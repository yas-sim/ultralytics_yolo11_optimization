import os
import time
import queue
import threading
from types import SimpleNamespace

import numpy as np
import cv2

import openvino as ov

from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.engine.results import Boxes
from ultralytics import YOLO            # Required only for downloading the model


def nms_cv2(output, conf_threshold=0.2, iou_threshold=0.8):
    """
    Postprocessing for YOLO model output
    :param output: model output (shape: (1, 84, 8400))
    :param input_shape: image size ([height, width])
    :param conf_threshold: threshold for confidences
    :param iou_threshold: threshould for IoU (for NMS)
    :return: bounding boxes and classes after NMS
    """
    # separate the bounding box coordinates (x, y, w, h) and class scores
    boxes = output[0, :4, :]  # (4, 8400)
    class_scores = output[0, 4:, :]  # (80, 8400)

    # Dequantization for fully int8-nized model    
    #boxes = ((boxes.astype(np.float32) + 128.0) / 256.0) * 640
    #class_scores = (class_scores.astype(np.float32) + 128.0) / 256.0

    # get the maximum class score and its index
    class_ids = np.argmax(class_scores, axis=0)  # class IDs of detections
    confidences = np.max(class_scores, axis=0)  # confidence values of detections

    # screen out low confidence boxes
    valid_indices = np.where(confidences > conf_threshold)[0]
    boxes = boxes[:, valid_indices]
    confidences = confidences[valid_indices]
    class_ids = class_ids[valid_indices]

    # convert the bouding box format (x_center, y_center, w, h) → (x0, y0, x1, y1)
    box_xywh = boxes.T
    box_xyxy = np.zeros_like(box_xywh)
    box_xyxy[:, 0] = box_xywh[:, 0] - box_xywh[:, 2] / 2  # x0
    box_xyxy[:, 1] = box_xywh[:, 1] - box_xywh[:, 3] / 2  # y0
    box_xyxy[:, 2] = box_xywh[:, 0] + box_xywh[:, 2] / 2  # x1
    box_xyxy[:, 3] = box_xywh[:, 1] + box_xywh[:, 3] / 2  # y1

    # apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(box_xyxy.tolist(), confidences.tolist(), conf_threshold, iou_threshold)
    if len(indices) > 0:
        indices = indices.flatten()

    # obtain NMS result
    final_boxes = box_xyxy[indices]
    final_confidences = confidences[indices]
    final_class_ids = class_ids[indices]

    return (final_boxes, final_confidences, final_class_ids)


def generate_ultralytics_box(boxes, confidences, class_ids,img_h=640, img_w=640):
    """ Generate ultralytics Boxes object from NMS results. Ultralytics tracker requires this object as input.
    :param boxes: (N, 4) array of bounding boxes in (x0, y0, x1, y1) format
    :param confidences: (N,) array of confidence values
    :param class_ids: (N,) array of class IDs
    :return: ultralytics Boxes object
    """
    data = np.concatenate([boxes, np.array([confidences]).T, np.array([class_ids]).T], axis=1)
    result = Boxes(boxes=data, orig_shape=(img_h, img_w))
    return result


tracker = BYTETracker(SimpleNamespace(**{
    'track_thresh': 0.5, 
    'track_buffer': 30, 
    'match_thresh': 0.8, 
    'frame_rate': 30, 
    'track_high_thresh':0.25,
    'track_low_thresh':0.1,
    'new_track_thresh':0.25,
    'fuse_score': True
}))

class Trace:
    def __init__(self, max_length=60):
        self.traces = dict()
        self.max_length = max_length

    def append(self, objid, point, clean=True):
        if clean:
            self.clean()
        if self.traces.get(objid) is None:
            self.traces[objid] = {'last_append': time.time(), 'points': [point]}
        else:
            self.traces[objid]['last_append'] = time.time()
            self.traces[objid]['points'].append(point)
            if len(self.traces[objid]['points']) > self.max_length:
                self.traces[objid]['points'].pop(0)

    def get_points(self, objid, clean=True):
        if clean:
            self.clean()
        if self.traces.get(objid) is None:
            return []
        return self.traces[objid]['points']

    def clean(self, timeout=20):
        current_time = time.time()
        for objid in list(self.traces.keys()):
            if current_time - self.traces[objid]['last_append'] > timeout:
                del self.traces[objid]


def pre_decode():
    frames = []
    cap = cv2.VideoCapture('people.mp4')
    while True:
        sts, img = cap.read()
        if sts == False or img is None:
            break
        frames.append(img)
    cap.release()
    return frames


def input_thread():
    global abort_flag
    global input_quque

    frames = pre_decode()   # Decode the input movie in advance
    nframe = 0
    while not abort_flag:
        img = frames[nframe]
        nframe += 1
        if len(frames) <= nframe:
            nframe = 0 
        img = cv2.resize(img, (640, 640))
        tensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0).astype(np.float32) / 255.0
        while input_quque.qsize() > 100 and abort_flag == False:       # Prevent excessive queue growth
            time.sleep(10e-3)
        input_quque.put((tensor, img))


def rendering_thread():
    global abort_flag
    global output_quque

    while abort_flag == False:
        while output_quque.empty() and abort_flag == False:
            time.sleep(0.01)
        if abort_flag:
            break
        img = output_quque.get()
        cv2.imshow('YOLO11n+OpenVINO+ByteTrack', img)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            abort_flag = True
    cv2.destroyAllWindows()


def prepare_openvino_model(device='GPU.1'):
    if not os.path.exists('./yolo11n_int8_openvino_model'):
        model = YOLO('yolo11n.pt', task='detect')
        model.export(format='openvino', int8=True, imgsz=640)
    model = ov.compile_model('yolo11n_int8_openvino_model/yolo11n.xml', device_name=device)
    return model


def main():
    global abort_flag
    global input_quque
    global output_quque

    input_quque = queue.Queue()
    output_quque = queue.Queue()

    trace = Trace(max_length=200)    # object tracking trace

    fps = 0
    fps_count = 10
    count = 0

    device = 'GPU.1'  # Inference device, can be 'GPU.0', 'GPU.1', 'CPU', 'NPU' etc. 'GPU.1' is an external GPU and 'GPU.0' is an internal GPU.
    model = prepare_openvino_model(device=device)  # Prepare and Load OpenVINO model

    input_th = threading.Thread(target=input_thread, daemon=True)
    rendering_th = threading.Thread(target=rendering_thread, daemon=True)
    input_th.start()
    rendering_th.start()

    stime = time.perf_counter()
    while abort_flag == False:
        while input_quque.empty() and abort_flag == False:
            time.sleep(10e-3)  # Wait for input queue to have data
        if abort_flag:
            break
        tensor, img = input_quque.get()

        res = model(tensor)[0]              # Inference with OpenVINO
        box, conf, cls = nms_cv2(res)       # NMS

        if len(conf) > 0:
            u_box = generate_ultralytics_box(box, conf, cls, img_h=640, img_w=640)
            tracks = tracker.update(u_box, img)
            for track in tracks:
                x0, y0, x1, y1, track_id = [ int(d) for d in track[:5] ]

                # Calculate the center of the bounding box for object tracking
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                trace.append(track_id, (cx, cy))

                conf = track[5]
                class_id = int(track[6])
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.polylines(img, [np.array(trace.get_points(track_id))], isClosed=False, color=(255, 255, 0), thickness=4)
                cv2.putText(img, f'ID:{track_id} {conf:.2f}',(x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Calculate FPS
        count += 1
        if count == fps_count:
            etime = time.perf_counter()
            inf_time = etime - stime
            fps = 1.0 / (inf_time / fps_count)
            print(f'*** fps={fps}')
            stime = etime
            count = 0

        text = f'fps={fps:6.2f} OpenVINO({device})+ByteTrack'
        cv2.putText(img, text, (0, 40), cv2.FONT_HERSHEY_PLAIN, 1.8, (0, 0, 0), 6)
        cv2.putText(img, text, (0, 40), cv2.FONT_HERSHEY_PLAIN, 1.8, (0, 255, 0), 2)

        output_quque.put(img)
    
    input_th.join()
    rendering_th.join()


if __name__ == "__main__":
    global abort_flag
    abort_flag = False
    main()
