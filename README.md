README.md

# Ultralytics YOLO11n + ByteTrack (Object Tracking) Optimization test

This is a repository of test codes for YOLO11n object detection and tracking code.  
The code optimize is done only for object detection part. ByteTrack algorithm are used as is without any optimization.  

|File Name|Using Ultralytics framework|Device|Description|FPS(reference)|
|---|---|---|---|---|
|yolo11n_ultralytics_cpu.py|✓|CPU|Run original Ultralytics YOLO11n with CPU|4|
|yolo11_cuda.py|✓|NV GPU|Used TensorRT to optimize. Still using Ultralytics framework|60|
|yolo11n_ultralytics_openvino.py||CPU/iGPU/NPU(/dGPU?)|Used OpenVINO to optimize. Still using Ultralytics framework|CPU 13 /iGPU 30|
|yolo11_ov_bytetracker.py||CPU/iGPU/NPU/dGPU|Use OpenVINO (without Ultralytics framework) to run object detection, and pass the result to ByteTrack to track the detected objects|CPU 28 / iGPU 60 / dGPU 72|
|yolo11_hailo_bytetrack||Hailo-8|Use Hailo-8 and HailoRT for object detection without Ultralytics framework|120|

- Benchmark HW1: Note PC, Dynabook SZ/LVL W6SZLV5FAL, Core i5-1235U + Hailo-8
- Benchmark HW2: Desktop PC, Core i7-10700K + RTX-3050 / Arc A380

- OS: Windows 11
- OpenVINO: 2025.2.0

## Note
- You need to download yolo11 model from Hailo Model Zoo ('`yolov11n.hef`')
- Hailo-8 is installed in a m.2-Thunderbolt4 box, and connected to the notebook PC via USB-C cable (Thunderbolt)
