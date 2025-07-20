
from collections import defaultdict
from ultralytics import YOLO
import os
import numpy as np
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import queue
import threading

import time

if not os.path.exists('./yolo11n_int8.engine'):
    model = YOLO('yolo11n.pt', task='detect')
    model.export(format='engine', int8=True, imgsz=640, simplify=True)
    os.rename('yolo11n.engine', 'yolo11n_int8.engine')

if not os.path.exists('./yolo11n.engine'):
    model = YOLO('yolo11n.pt', task='detect')
    model.export(format='engine', half=True, simplify=True, imgsz=640)

model = YOLO('yolo11n_int8.engine')

cap = None
key = -1
track_history = defaultdict(lambda: [])
prev_time = 0
count = 0
fps_count = 10
fps = 0

stime = time.perf_counter()

def display_frame_thread():
    global abort_flag
    global output_quque
    while not abort_flag:
        while output_quque.empty() and not abort_flag:
            time.sleep(0.01)
        if abort_flag:
            break
        img = output_quque.get()
        cv2.imshow('YOLO11n+Ultralytics(TRT i8)', img)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            abort_flag = True
    cv2.destroyAllWindows()

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

nframe = 0
frames = pre_decode()   # Decode the input movie in advance
output_quque = queue.Queue()
abort_flag = False

output_th = threading.Thread(target=display_frame_thread, daemon=True)
output_th.start()

while key != 27 and not abort_flag:

    img = frames[nframe]
    nframe += 1
    if len(frames) <= nframe:
        nframe = 0 
    img = cv2.resize(img, (640, 640))

    results = model.track(img, device='cuda', tracker='bytetrack.yaml', persist=True, verbose=False)
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

    text = f'fps={fps:6.2f} Ultralytics CUDA (TensorRT i8)'
    cv2.putText(frame, text, (0, 40), cv2.FONT_HERSHEY_PLAIN, 1.8, (0, 0, 0), 6)
    cv2.putText(frame, text, (0, 40), cv2.FONT_HERSHEY_PLAIN, 1.8, (0, 255, 0), 2)

    if not output_quque.full():
        output_quque.put(frame)

    count += 1
    if count == fps_count:
        etime = time.perf_counter()
        inf_time = etime - stime
        fps = 1.0 / (inf_time / fps_count)
        print(f'*** fps={fps}')
        stime = etime
        count = 0

output_th.join()
