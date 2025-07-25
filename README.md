README.md

# Ultralytics YOLO11n + ByteTrack (Object Tracking) Optimization test

This is a repository of test codes for YOLO11n object detection and tracking code.  
The code optimize is done only for object detection part. ByteTrack algorithm are used as is without any optimization.  

|File Name|Using Ultralytics framework|Device|Description|FPS(reference)|
|---|:---:|:---:|---|:---:|
|`yolo11n_ultralytics_cpu.py`|✓|CPU|Run original Ultralytics YOLO11n with CPU|4|
|`yolo11n_ultralytics_openvino.py`|✓|CPU/iGPU/NPU(/dGPU?)|Used OpenVINO to optimize. Still using Ultralytics framework|CPU 13<br>iGPU 30|
|`yolo11_ultralytics_cuda.py`|✓|NV GPU|Used TensorRT to optimize. Still using Ultralytics framework|60|
|`yolo11_ov_bytetracker.py`||CPU/iGPU/NPU/dGPU|Use OpenVINO (without Ultralytics framework) to run object detection, and pass the result to ByteTrack to track the detected objects|CPU 28<br>iGPU 60<br>dGPU 72|
|`yolo11_hailo_bytetrack.py`||Hailo-8|Use Hailo-8 and HailoRT for object detection without Ultralytics framework|120|

- Benchmark HW1: Note PC, Dynabook SZ/LVL W6SZLV5FAL, Core i5-1235U + Hailo-8
- Benchmark HW2: Desktop PC, Core i7-10700K + RTX-3050 / Arc A380

- OS: Windows 11
- OpenVINO: 2025.2.0

## How to setup the environment

1. Install required Python modules

### Windows
```sh
python -m venv venv
venv\Scripts\activate
python -m pip install -U pip
pip install -U setuptools wheel
pip install -r requirements.txt
```
### Linux
```sh
python3 -m venv venv
. venv/bin/activate
python3 -m pip install -U pip
pip install -U setuptools wheel
pip install -r requirements.txt
```
*Note:* 
- You need to install TensorRT for CUDA/TRT version, and HailoRT for Hailo-8 version. Please go and refer to the vendor's web documents for the details.  

- You need to download yolo11 model from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) ('[`yolov11n.hef`](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.16.0/hailo8/yolov11n.hef)')

## Movie

[!['Youtube demo movie'](./resources/youtube_thumbnail.jpg)](https://www.youtube.com/watch?v=ID7BPbTEiI4)

## Remarks
- Hailo-8 is installed in a m.2-Thunderbolt4 box, and connected to the notebook PC via USB-C cable (Thunderbolt) in my test
