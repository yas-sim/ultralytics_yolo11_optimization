import cv2
import numpy as np
import time
import openvino as ov
from types import SimpleNamespace
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.engine.results import Boxes

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
    """ Generate ultralytics Boxes object from NMS results
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

cap = None
key = -1
stime = time.perf_counter()
trace = Trace(max_length=60)

fps = 0
fps_count = 10
count = 0

device = 'CPU'
model = ov.compile_model('yolo11n_int8_openvino_model/yolo11n.xml', device_name=device)

while key != 27:
    if cap is None:
        cap = cv2.VideoCapture('people.mp4')
    sts, img = cap.read()
    if sts == False or img is None:
        cap.release()
        cap = None
        continue
    img = cv2.resize(img, (640, 640))
    tensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0).astype(np.float32) / 255.0

    res = model(tensor)[0]
    box, conf, cls = nms_cv2(res)
    res = generate_ultralytics_box(box, conf, cls, img_h=640, img_w=640)

    if len(res.conf) > 0:
        tracks = tracker.update(res, img)
        for track in tracks:
            x0, y0, x1, y1, track_id = [ int(d) for d in track[:5] ]
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            trace.append(track_id, (cx, cy))

            conf = track[5]
            class_id = int(track[6])
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.polylines(img, [np.array(trace.get_points(track_id))], isClosed=False, color=(230, 230, 230), thickness=2)
            cv2.putText(img, f'ID:{track_id}',(x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
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

    cv2.imshow('a', img)
    key = cv2.waitKey(1)

