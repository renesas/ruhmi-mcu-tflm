# Quick Deploy

Program the prebuilt EK-RA8P1 images with **Intel HEX + Renesas Flash Programmer (RFP)**. No e² studio or build is required.

## Available examples

Models run on the TFLM interpreter with Ethos-U55 NPU acceleration. The gesture classifier in the combined hand example uses FP32 on the CPU. Each deployment folder also contains an example-specific README.

| Use case | Model | Devices Supported | Model Reference | Binary | Source project (.zip) |
| --- | --- | --- | --- | --- | --- |
| Face detection | YOLO-fastest-192 | EK-RA8P1 | [emza-vs/ModelZoo](https://github.com/emza-vs/ModelZoo/tree/master/object_detection) | [Download HEX](EK_RA8P1_face_detection_yolo_fastest_TFLM_fsp_6_6_0/ek_ra8p1_vision_face_detection_yolo_fastest_TFLM_FSP660.hex) | [Download](../ek_ra8p1_vision_face_detection_yolo_fastest_TFLM_FSP660.zip) |
| Image classification | MobileNet V1 | EK-RA8P1 | [tensorflow/models (MobileNet V1)](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md) | [Download HEX](EK_RA8P1_image_classification_mobilenet_v1_TFLM_fsp_6_6_0/ek_ra8p1_vision_image_classification_mobilenet_v1_TFLM_FSP660.hex) | [Download](../ek_ra8p1_vision_image_classification_mobilenet_v1_TFLM_FSP660.zip) |
| Palm detection | Palm Detection INT8 | EK-RA8P1 | [hand-gesture-recognition-using-onnx](https://github.com/PINTO0309/hand-gesture-recognition-using-onnx) | [Download HEX](EK_RA8P1_palm_detection_TFLM_fsp_6_6_0/ek_ra8p1_vision_palm_detection_model_camera_LCD_TFLM_FSP660.hex) | - |
| Palm detection + Hand landmark | Palm Detection INT8 + Hand Landmark INT8 | EK-RA8P1 | [hand-gesture-recognition-using-onnx](https://github.com/PINTO0309/hand-gesture-recognition-using-onnx) | [Download HEX](EK_RA8P1_palm_detection_hand_landmark_TFLM_fsp_6_6_0/ek_ra8p1_vision_palm_detection_hand_landmarkmodel_camera_LCD_TFLM_FSP660.hex) | - |
| Palm detection + Hand landmark + Gesture recognition | Palm Detection INT8 + Hand Landmark INT8 + Keypoint Classifier FP32 | EK-RA8P1 | [hand-gesture-recognition-using-onnx](https://github.com/PINTO0309/hand-gesture-recognition-using-onnx) | [Download HEX](EK_RA8P1_palm_detection_hand_landmark_gesture_recognition_TFLM_fsp_6_6_0/ek_ra8p1_vision_palm_detection_hand_landmarkmodel_gesture_recognition_camera_LCD_TFLM_FSP660.hex) | [Download](../ek_ra8p1_vision_palm_detection_hand_landmarkmodel_gesture_recognition_camera_LCD_TFLM_FSP660.zip) |

> Refer to each published example's README for board-test results.

## Hardware and software

- Connect the camera, LCD and power as shown in the [setup overview](../README.md#connecting-and-powering-up-the-ek-ra8p1-board). Set **SW4 = all OFF** and connect the PC to **DEBUG1 USB-C (J10)**.
- Download and install [Renesas Flash Programmer (RFP)](https://www.renesas.com/en/software-tool/renesas-flash-programmer-programming-gui), selecting the package suitable for your environment with RA8P1 and J-Link support.
- Close J-Link Commander and active e² studio debug sessions to release the probe.

## Program a HEX image with RFP

### 1. Create a project and connect

Select **File > Create a new project**. Choose **Microcontroller = RA**, **Tool = J-Link**, **Interface = SWD**, and a project name/location. Select **Connect** and confirm **R7KA8P1KFLCAC**. If multiple probes are present, select the intended on-board J-Link.

### 2. Set external memory only when needed

Use the table below to determine whether to check **Enable external memory settings** on the **External Memory Settings** tab. Do not use another board's loader.

| Use case | External Memory Settings |
| --- | --- |
| Face detection | Leave external memory disabled. |
| Image classification | Check **Enable external memory settings**, choose **Target select = EK-RA8P1**, and confirm the loader covers `0x90000000`–`0x90073A8F`. |
| Palm detection | Check **Enable external memory settings**, choose **Target select = EK-RA8P1**, and confirm the loader covers `0x90000000`–`0x9014A1FF`. |
| Palm detection + Hand landmark | Check **Enable external memory settings**, choose **Target select = EK-RA8P1**, and confirm the loader covers `0x90000000`–`0x9024E5DF`. |
| Palm detection + Hand landmark + Gesture recognition | Check **Enable external memory settings**, choose **Target select = EK-RA8P1**, and confirm the loader covers `0x90000000`–`0x9024E5DF`. |

### 3. Add the image and check the security boundary

1. On the **Operation** tab, select **Add/Remove Files...** and add the `.hex` with **address offset = 0**.
2. From the menu bar, select **Target Device > Read Flash Options** to read the current device configuration. Check the **Code Secure** size in the readback results.
3. For the supplied **Secure/Flat CPU0 examples**, confirm **Code Secure = 1024 KB**. If it is already 1024 KB, no boundary change is needed; continue to step 4 below. Otherwise, back up important data and complete [Security boundary setup](#security-boundary-setup) before programming.

### 4. Check programming and erase settings

Set **Operation Settings** as follows.

| State | Options |
| --- | --- |
| Enabled | Program, Verify, Erase before Program |
| Disabled | Erase (separate command), Fill with 0xFF, Program Flash Options |

Set **Block Settings** as follows.

| Region | Selection |
| --- | --- |
| Code MRAM 1 (`0x02000000`), Config Areas 1/2 | Select for the supplied HEX images |
| Code MRAM 2 (`0x12000000`), Config Area 3, Config OTP Areas | Do not select |
| External OSPI | **Image classification**: `0x90000000`–`0x90073A8F`<br>**Palm detection**: `0x90000000`–`0x9014A1FF`<br>**Palm detection + Hand landmark**: `0x90000000`–`0x9024E5DF`<br>**Palm detection + Hand landmark + Gesture recognition**: `0x90000000`–`0x9024E5DF`. |

Code MRAM 1/2 are Secure/Non-Secure mappings of the same memory. Do not modify Lockbit or OTP.

> Back up important data: **Erase before Program** can erase other data in the same erase block. Do not use chip/all-block erase, device initialization or security-state changes.

### 5. Program, verify and run

1. On **Operation**, check the HEX file and operation summary, then select **Start**. Keep USB and power connected until completion.
2. Confirm **Program + Verify** succeed without ignored-data or address-range warnings.
3. Close RFP, reset the board (or fully power-cycle it), and check the camera image and AI results on the LCD.

## Security boundary setup

Use only for the supplied **Secure/Flat CPU0 examples**. Skip the change if Code Secure is already **1024 KB**; do not apply it to applications requiring a Non-Secure partition.

1. Back up important data. Read and record the current configuration via **Target Device > Read Flash Options**.
2. On **Flash Options > Boundary**, apply the following settings. Leave **all other groups at Do Nothing**, including DLM/keys, Security, CPU, OTP, Anti-Rollback and certificates.

   | Setting | Value |
   | --- | --- |
   | Set Option | Set |
   | Use Renesas Partition Data File | No |
   | Code Secure Size [KB] | 1024 |

3. On **Operation Settings**, enable **only Program Flash Options**; disable Erase, Program and Verify. Select **Start** on Operation. **Stop** if asked to initialize, erase or change the security state.
4. After success, reset as instructed, reconnect and read back **Code Secure = 1024 KB**.
5. Set **Boundary > Set Option = Do Nothing**, then restore the [ordinary programming settings](#4-check-programming-and-erase-settings). Do Nothing leaves the programmed boundary unchanged.

If you later use e² studio, **Debugger > Connection Settings > TrustZone > Set TrustZone secure/non-secure boundaries = Yes** can resize the partition. Set it to **No** in each relevant debug configuration only after establishing a suitable shared partition.

## Build a HEX image in e² studio

**Optional: skip this section when using the prebuilt HEX.**

1. Extract the source ZIP and import via **File > Import > Rename & Import Existing C/C++ Project Into Workspace**.
2. Open `configuration.xml` and select **Generate Project Content**.
3. In **Properties > C/C++ Build > Settings**, select **Debug** (or the required configuration). Set **Tool Settings > Objcopy > General > OutFormat = Intel Hex**, then **Apply and Close**.
4. Clean and build; confirm `llvm-objcopy ... -O ihex ...` succeeds. Use the `.hex` in the active `Debug` or `Release` folder with [RFP](#program-a-hex-image-with-rfp).

HEX output does not change memory placement or security boundaries. If you change model placement, rebuild and check the HEX addresses and loader selection.

## Notes

- **Address error (`E1000008`):** check the security boundary. For unsupported-memory or ignored-data warnings, stop and inspect the HEX/loader; do not relocate addresses, disable range checks or erase the chip to bypass errors.
- **Black screen after programming:** close RFP and reset/power-cycle first. If it persists, check startup and SRAM security; Code Secure alone does not configure the application's complete RAM security.
- Use the supplied HEX for the selected example. HEX file size is not MRAM usage.
