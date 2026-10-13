# AeroWatch AI — Comprehensive Engineering Audit Report

**Date:** 2026-10-12  
**System:** Windows | Python 3.14 (.venv) | NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM) | CUDA 12.8 / PyTorch 2.14.1+cu132 | Ultralytics 8.4.175  

---

## 1. Executive Summary & Root Cause Analysis

A systematic audit of the AeroWatch AI codebase was conducted across 10 critical operational dimensions. The audit identified the exact root causes of class name instability, camera reliability failures, and video processing bottlenecks:

1. **Inconsistent Class Mapping & COCO/Custom Collision:**
   - In `app.py`, `CANONICAL_CLASSES` was hardcoded to 6 classes (`0: drone, 1: aircraft, 2: bird, 3: balloon, 4: military_jet, 5: kamikaze_drone`).
   - In COCO 80-class models (`yolo26n.pt`), class `4` is `airplane`, class `14` is `bird`. If code falls back to `CANONICAL_CLASSES[4]`, a COCO airplane is misidentified as a `military_jet` (Hostile Warplane).
   - UI code contained heuristics (`if "military" in cname or "jet" in cname`) that attempted to infer military warplane subtypes from generic aircraft bounding boxes.
   - In civilian aerial monitoring, the canonical standard requires:
     - `drone`
     - `airplane` internally, displayed consistently as **`Aircraft`**
     - `bird`
     - `balloon`

2. **Camera Lifecycle & Capture Fragility:**
   - `app.py` previously relied on Streamlit's `st.camera_input`, which is a static browser-photo widget rather than a real-time OpenCV camera stream.
   - There was no camera manager with a defined state machine (`STOPPED -> OPENING -> RUNNING -> ERROR -> STOPPED`).
   - No backend probing (`cv2.CAP_DSHOW` vs `cv2.CAP_MSMF`), no validation of opened streams against empty frames, and no frame-buffer limit (`CAP_PROP_BUFFERSIZE=1`), leading to capture latency or device lockouts.

3. **Video Processing Pipeline Bottlenecks:**
   - Every inference call loaded or instantiated `model = YOLO(...)` without session caching or persistent worker reuse.
   - Inference executed in default FP32 without warm-up or CUDA FP16 half-precision (`half=True`), which is fully supported by the RTX 3050 Tensor Cores.
   - Video processing lacked stage-by-stage profiling (decode, preprocess, inference, postprocess, render, write).
   - Video scrubbers in the UI repeatedly opened and closed `cv2.VideoCapture` per slider movement.

4. **Codec and I/O Handling:**
   - Legacy OpenCV scripts wrote `mp4v` (MPEG-4 Part 2), causing web browser playback failure (black screen at 0:00).
   - Need guaranteed H.264 encoding with automated verification that saved output files can be reopened and decoded.

---

## 2. Codebase Audit Checklist (10 Dimensions)

| Dimension | Current State | Root Cause & Risk | Required Engineering Fix |
| :--- | :--- | :--- | :--- |
| **1. Entry Point** | Monolithic `app.py` (716 lines) mixing UI, model loading, and cv2 calls | High coupling, race conditions on Streamlit reruns | Modularize into `model_manager`, `camera_manager`, `video_processor`, `diagnostics` |
| **2. Model Selection** | Raw file scanning without metadata introspection | Broken fallback to COCO; no clear indicator of custom vs pretrained capabilities | Dedicated `ModelManager` with metadata inspection, class vocabulary normalization, and explicit COCO mode |
| **3. Live Camera** | Browser snapshot widget (`st.camera_input`) | No streaming; cannot select Windows camera backend (DirectShow/MSMF) or resolution | Robust `CameraManager` with background capture thread, `BUFFERSIZE=1`, and graceful teardown |
| **4. Video Pipeline** | Synchronous frame-by-frame loop without stage timers | Low FPS (~12-18 FPS), no latency telemetry, UI freezing | Multi-stage profiled `VideoProcessor`, FP16 half-precision, frame-skipping options |
| **5. Video Output** | Codec mismatches, unverified file writes | Browser playback failures, potential zero-byte files | Enforce H.264 via `imageio-ffmpeg` or validated OpenCV writers; test reopening |
| **6. Streamlit Lifecycle** | Re-instantiating models and capture objects on every rerun | Memory leaks, camera device busy errors | Use `st.session_state` and `@st.cache_resource` with atomic locks |
| **7. Class Mapping** | Mixed custom labels, COCO IDs, and invented military subtypes | Aircraft labeled as warplanes, class confusion | Normalize all civilian outputs to `drone`, `airplane` (displayed as `Aircraft`), `bird`, `balloon` |
| **8. Hardware Init** | Inconsistent CUDA/CPU checks; default FP32 | Under-utilizing RTX 3050 Laptop GPU (4 GB VRAM) | CUDA device 0 with FP16 `half=True` inference and one-shot warmup |
| **9. Concurrency & Queues**| Unbounded or missing queues | UI freezes and frame lag | Bounded latest-frame buffer (size=1) for live feeds |
| **10. Dependencies** | Python 3.14, PyTorch 2.14.1+cu132, Ultralytics 8.4.175 | Clean environment; need unified test harness | Add unit test suite in `tests/` with mocked camera and video pipelines |

---

## 3. Immediate Action Plan

1. **`model_manager.py`**: Model loading, metadata reading, strict separation of Pretrained COCO vs Custom Checkpoint, canonical class normalization, CUDA/FP16 warmup.
2. **`camera_manager.py`**: Threaded camera capture, Windows backend fallback (`CAP_DSHOW` -> `CAP_MSMF` -> Default), state machine, error diagnostics.
3. **`video_processor.py`**: High-performance video detection pipeline with stage-by-stage profiling, bounded queues, and verified H.264 writing.
4. **`diagnostics.py`**: Hardware, camera, video codec, and model diagnostic routines.
5. **`tests/`**: Unit tests for class mapping, camera lifecycle, video processing, and integration smoke test with synthetic video.
6. **`app.py`**: Clean, modular Streamlit UI with 7 dedicated tabs (Image Detection, Live Camera, Video-File Detection, Model Information, Performance Statistics, Diagnostics & Logs, Dataset & Validation).
7. **`README_AR.md`**: Updated documentation in Egyptian Arabic with exact Windows PowerShell commands.
