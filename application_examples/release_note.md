Date: 2026-10-27

# Palm detection TFLM Quick Deploy release

- Added the Palm detection `.hex` file (HEX-only) to [quick_deploy](quick_deploy/README.md) and updated the related documentation.
- External OSPI programming is required for `0x90000000`–`0x9014A1FF`. Select the EK-RA8P1 loader and follow the common RFP programming instructions.

---

Date: 2026-10-27

# Palm detection + Hand landmark TFLM Quick Deploy release

- Added the Palm detection + Hand landmark `.hex` file (HEX-only) to [quick_deploy](quick_deploy/README.md) and updated the related documentation.
- External OSPI programming is required for `0x90000000`–`0x9024E5DF`. Select the EK-RA8P1 loader and follow the common RFP programming instructions.

---

Date: 2026-10-21

# Palm detection + Hand landmark + Gesture recognition TFLM runtime migration + Quick Deploy folder update

## Source project

- Published the complete TFLM [source project ZIP](ek_ra8p1_vision_palm_detection_hand_landmarkmodel_gesture_recognition_camera_LCD_TFLM_FSP660.zip) and [application documentation](palm_detection_hand_landmark_gesture_recognition/README.md) for Palm detection + Hand landmark + Gesture recognition.
- Changed the inference time shown on the LCD in this example to the pipeline time of the latest detection frame.

## Quick Deploy

- Added the Palm detection + Hand landmark + Gesture recognition `.hex` file to [quick_deploy](quick_deploy/README.md) and updated the [related documentation](quick_deploy/EK_RA8P1_palm_detection_hand_landmark_gesture_recognition_TFLM_fsp_6_6_0/README.md).
- External OSPI programming is required for `0x90000000`–`0x9024E5DF`. Select the EK-RA8P1 loader and follow the common RFP programming instructions.

---

Date: 2026-10-09

# Image Classification TFLM runtime migration + Quick Deploy folder creation

## Source project

- Published the complete TFLM [source project ZIP](ek_ra8p1_vision_image_classification_mobilenet_v1_TFLM_FSP660.zip) and [application documentation](image_classification/README.md) for Image classification.

## Quick Deploy

- Added the [quick_deploy folder](quick_deploy/README.md), containing `.hex` files for quick deployment of Face detection and Image classification, along with detailed usage instructions.

---

Date: 2026-10-05

# Face Detection TFLM runtime migration

- Published the complete TFLM [source project ZIP](ek_ra8p1_vision_face_detection_yolo_fastest_TFLM_FSP660.zip) and [application documentation](face_detection/README.md) for Face detection.

---

Date: 2026-09-04

# 5 application examples (FSP v6.6.0 release update).  

### Software requirements

  - e² studio 2026-07  
  - FSP 6.6.0  
  - LLVM 22.1.0  

## Overall

- Unified all five demo branches to the FSP v6.6.0 baseline for consistent build/import behavior in the latest e² studio environment.  
- Improved runtime stability and visual consistency across demos, especially for overlay behavior, tracking continuity, and frame-to-frame smoothness.
- Resolved reliability issues in inference input/output handling and display update flow to reduce intermittent mis-detections or UI artifacts.  

### 1) Face Detection  

- Updated project baseline to FSP v6.6.0 for release alignment.  
- Improved bounding-box stability so stationary faces show significantly less box flicker while moving faces remain responsive.  
- Stabilized per-frame processing cost by limiting pre-NMS candidate volume to the app display capacity.  
- Fixed output quantization handling to prevent rare cases where inference results could become inconsistent across builds/toolchains.  

### 2) Image Classification  

- Updated project baseline to FSP v6.6.0 for release alignment.  
- Improved Top-N result stability so near-tie classes are less likely to swap rapidly between adjacent frames.  
- Improved sidebar label/probability rendering consistency to avoid occasional truncation or stale text artifacts.  
- Fixed output/result handling robustness to keep classification output behavior stable under continuous real-time operation.  

### 3) Palm Detection

- Updated project baseline to FSP v6.6.0 for release alignment.  
- Fixed a display update race that could intermittently clear overlays mid-cycle, seen as brief box blinking.  
- Improved box stability during hand motion and still-hand scenarios by adding temporal smoothing with slot-aware matching.  

### 4) Palm Detection + Hand Landmark  

- Updated project baseline to FSP v6.6.0 for release alignment.  
- Improved overlay consistency by using landmark-derived hand geometry and by correcting edge-of-screen clipping behavior.  
- Reduced AI pipeline time by switching to tracking-first operation (detector runs periodically instead of every frame when tracking is stable).  
- Improved visual smoothness and interaction stability by applying temporal filtering to both hand ROI updates and landmark point rendering, reducing frame-to-frame jitter while keeping hand motion responsive.
