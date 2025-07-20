
from collections import defaultdict
from ultralytics import YOLO
import os
import numpy as np
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

import time

model = YOLO('yolo11n.pt', task='detect').to('cpu')

cap = None
key = -1
track_history = defaultdict(lambda: [])
prev_time = 0
count = 0
fps_count = 10
fps = 0

stime = time.perf_counter()


while key != 27:
    if cap is None:
        cap = cv2.VideoCapture('people.mp4')
    sts, img = cap.read()        
    if sts == False or img is None:
        cap.release()
        cap = None
        continue

    img = cv2.resize(img, (640, 640))

    a = model.predict(img)
    results = model.track(img, device='cpu', tracker='bytetrack.yaml', persist=True, verbose=False)
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

        text = f'fps={fps:6.2f} Ultralytics CPU'
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
