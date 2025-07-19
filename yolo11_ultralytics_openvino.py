
from collections import defaultdict
from ultralytics import YOLO
import os
import numpy as np
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

import time

if not os.path.exists('./yolo11n_openvino_model'):
    model = YOLO('yolo11n.pt', task='detect')
    model.export(format='openvino', half=True, imgsz=640)

if not os.path.exists('./yolo11n_int8_openvino_model'):
    model = YOLO('yolo11n.pt', task='detect')
    model.export(format='openvino', int8=True, imgsz=640)

#model = YOLO('yolo11n_openvino_model/', task='detect')
model = YOLO('yolo11n_int8_openvino_model/', task='detect')

cap = None
key = -1
track_history = defaultdict(lambda: [])
prev_time = 0
count = 0
fps_count = 10
fps = 0

stime = time.perf_counter()

device = 'CPU'  # Inference device, can be 'GPU.0', 'GPU.1', 'CPU', 'NPU' etc. 'GPU.1' is an external GPU and 'GPU.0' is an internal GPU.

while key != 27:
    if cap is None:
        cap = cv2.VideoCapture('people.mp4')
    sts, img = cap.read()        
    if sts == False or img is None:
        cap.release()
        cap = None
        continue

    img = cv2.resize(img, (640, 640))
    
    results = model.track(img, device=f'intel:{device}', tracker='bytetrack.yaml', persist=True, verbose=False)
    result = results[0]
    if result.boxes and result.boxes.is_track:
        boxes = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()

        frame = result.plot()

        for (x, y, w, h), track_id in zip(boxes, track_ids):
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 60:
                track.pop(0)
            
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=4)

    text = f'fps={fps:6.2f} Ultralytics(OpenVINO) {device}'
    cv2.putText(frame, text, (0, 40), cv2.FONT_HERSHEY_PLAIN, 1.8, (0, 0, 0), 6)
    cv2.putText(frame, text, (0, 40), cv2.FONT_HERSHEY_PLAIN, 1.8, (0, 255, 0), 2)


    cv2.imshow('result', frame)
    key = cv2.waitKey(1)

    count += 1
    if count == fps_count:
        etime = time.perf_counter()
        inf_time = etime - stime
        fps = 1.0 / (inf_time / fps_count)
        print(f'*** fps={fps}')
        stime = etime
        count = 0

cv2.destroyAllWindows()
cap.release()
