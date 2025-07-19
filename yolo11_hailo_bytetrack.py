import os
import time
import queue
import threading
from functools import partial
from types import SimpleNamespace

import numpy as np
import cv2

from hailo_platform import HailoSchedulingAlgorithm, VDevice, Device

from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.engine.results import Boxes
from ultralytics import YOLO            # Required only for downloading the model


def generate_ultralytics_box(res, img_h=640, img_w=640):
    bboxes = []
    for clsid, detections in enumerate(res):
        if len(detections) == 0: continue
        for det in detections:
            bbox, conf = det[:4], det[4]
            if conf < 0.6 or conf >= 1.0:
                continue
            x0 = int(bbox[1] * img_w)
            y0 = int(bbox[0] * img_h)
            x1 = int(bbox[3] * img_w)
            y1 = int(bbox[2] * img_h)
            bboxes.append([x0, y0, x1, y1, conf, clsid])

    bboxes = np.array(bboxes)
    result = Boxes(boxes=bboxes, orig_shape=(img_h, img_w))
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
    cap = None

    frames = pre_decode()   # Decode the input movie in advance
    nframe = 0
    while not abort_flag:
        img = frames[nframe]
        nframe += 1
        if len(frames) <= nframe:
            nframe = 0 
        img = cv2.resize(img, (640, 640))
        tensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(tensor)
        while input_quque.qsize() >= 100:       # Prevent excessive queue growth
            time.sleep(10e-3)
        input_quque.put((tensor, img))
    cap.release()


def rendering_thread():
    global abort_flag
    global output_quque

    while abort_flag == False:
        while output_quque.empty():
            time.sleep(0.01)
        img = output_quque.get()
        cv2.imshow('YOLO11n+Hailo8+ByteTrack', img)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            abort_flag = True
    cv2.destroyAllWindows()



def postprocess_thread():
    global abort_flag
    global predict_queue

    trace = Trace(max_length=200)    # object tracking trace

    count= 0
    fps = 0
    fps_count = 10

    stime = time.perf_counter()

    while abort_flag == False:
        while predict_queue.empty():
            time.sleep(10e-3)
        img, res = predict_queue.get()
        u_box = generate_ultralytics_box(res, 640, 640)

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

        text = f'fps={fps:6.2f} Hailo-8 + ByteTrack'
        cv2.putText(img, text, (0, 40), cv2.FONT_HERSHEY_PLAIN, 1.8, (0, 0, 0), 6)
        cv2.putText(img, text, (0, 40), cv2.FONT_HERSHEY_PLAIN, 1.8, (0, 255, 0), 2)

        output_quque.put(img)


def callback(bindings, img, completion_info):
    global predict_queue
    if completion_info.exception:
        print(f'Inference error: {completion_info.exception}')
        return
    for binding in bindings:
        res = binding.output().get_buffer()
        predict_queue.put((img, res))



def main():
    global abort_flag
    global input_quque
    global output_quque
    global predict_queue

    input_quque = queue.Queue()
    output_quque = queue.Queue()
    predict_queue = queue.Queue()

    trace = Trace(max_length=200)    # object tracking trace

    fps = 0
    fps_count = 10
    count = 0

    input_th = threading.Thread(target=input_thread, daemon=True)
    rendering_th = threading.Thread(target=rendering_thread, daemon=True)
    postprocess_th = threading.Thread(target=postprocess_thread, daemon=True)

    input_th.start()
    rendering_th.start()
    postprocess_th.start()

    hailo8_devices = Device.scan()
    print(f'{len(hailo8_devices)} Hailo8 devices are found. {hailo8_devices}')
    params = VDevice.create_params()
    params.device_ids = hailo8_devices
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

    with VDevice(params) as vdevice:

        # Create an infer model from an HEF:
        infer_model = vdevice.create_infer_model('yolov11n.hef')
        model_input = infer_model.input()
        print(model_input.name, model_input.shape)

        # Configure the infer model and create bindings for it
        with infer_model.configure() as configured_infer_model:
            bindings = configured_infer_model.create_bindings()
            buffer = np.zeros(infer_model.output().shape).astype(np.float32)
            bindings.output().set_buffer(buffer)

            # Run asynchronous inference
            queue_size = configured_infer_model.get_async_queue_size()
            print(f'Async queue size: {queue_size}')

            stime = time.perf_counter()
            while abort_flag == False:
                while input_quque.empty() and abort_flag == False:
                    time.sleep(10e-3)
                if abort_flag:
                    break
                tensor, img = input_quque.get()

                bindings.input().set_buffer(tensor)

                configured_infer_model.wait_for_async_ready()
                job = configured_infer_model.run_async(
                    bindings = [bindings], 
                    callback = partial(callback, [bindings], img)
                )
    
    input_th.join()
    rendering_th.join()
    postprocess_th.join()


if __name__ == "__main__":
    global abort_flag
    abort_flag = False
    main()
