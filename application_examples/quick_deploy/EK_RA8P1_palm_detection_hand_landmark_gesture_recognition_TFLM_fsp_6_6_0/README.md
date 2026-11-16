# Palm Detection + Hand Landmark + Gesture Recognition (TFLM) - EK-RA8P1

The camera feeds a three-stage pipeline: palm detection, 21-point hand landmark extraction, and FP32 gesture classification. The LCD displays up to two hands with skeleton overlays, handedness, and Open / Close / Pointer labels.

## Files

| File | Size | Description |
| --- | --- | --- |
| [HEX image](ek_ra8p1_vision_palm_detection_hand_landmarkmodel_gesture_recognition_camera_LCD_TFLM_FSP660.hex) | 7,423,736 bytes | Select this image in Renesas Flash Programmer |

## Build information

| Item | Value |
| --- | --- |
| Devices Supported | EK-RA8P1 |
| e² studio / FSP | 2026-07 / 6.6.0 |
| Toolchain | LLVM/Clang (ATfE 22.1.0) |
| Build configuration | e² studio debug |
| Runtime | TFLM + Ethos-U55; gesture classification uses FP32 on the CPU |
| Programming | Intel HEX + Renesas Flash Programmer (RFP) |
| Source project (.zip) | [Download](../../ek_ra8p1_vision_palm_detection_hand_landmarkmodel_gesture_recognition_camera_LCD_TFLM_FSP660.zip) |

## Memory allocation

| Item | Value |
| --- | --- |
| Palm model data | 1,352,192 bytes, OSPI Flash load image copied to SDRAM |
| Landmark model data | 1,065,952 bytes, OSPI Flash load image copied to SDRAM |
| Gesture model data | 6,224 bytes, internal MRAM |
| Shared non-persistent tensor arena | 1,008,640 bytes, on-chip SRAM |
| Persistent tensor arenas | 4,096 bytes per vision model, 8,192 bytes total, on-chip SRAM |
| Gesture tensor arena | 4,096 bytes, on-chip SRAM |
| Internal MRAM image span | About 216 KiB of 1 MiB; `0x02000000`–`0x0203600B` (with gaps) |
| OSPI model addresses | `0x90000000`–`0x9024E5DF` |

The HEX also includes startup option words in Config Areas 1/2. It contains no SRAM or SDRAM address records; startup copies the palm and landmark models from OSPI to SDRAM.

The two vision models run sequentially and share the non-persistent tensor arena, while retaining separate persistent tensor arenas. The gesture classifier uses an independent tensor arena in the display thread. These tensor allocations total 1,020,928 bytes of on-chip SRAM, excluding other application buffers and stacks.

## Program and run

Follow [Quick Deploy](../README.md#program-a-hex-image-with-rfp). Enable **External Memory Settings**, select **EK-RA8P1**, and ensure the OSPI loader covers `0x90000000`–`0x9024E5DF`. Confirm **Code Secure = 1024 KB** before programming.

After successful writing and verification, close RFP and reset or power-cycle the board. Confirm the camera image, hand skeletons, handedness, and gesture labels on the LCD. **SW2** pauses or resumes the pipeline.

The LCD shows the pipeline time of the latest completed detection frame and retains it during tracking-only frames. It is not pure model inference time or end-to-end latency. See the application README for the supplied performance figures.
