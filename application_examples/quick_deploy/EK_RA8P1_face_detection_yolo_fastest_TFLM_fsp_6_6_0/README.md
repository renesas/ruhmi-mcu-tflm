# Face Detection (YOLO-fastest-192, TFLM) - EK-RA8P1

A camera captures images, TFLM runs the model on the Ethos-U55 NPU, and the LCD shows bounding boxes, the face count and inference time.

## Files

| File | Size | Description |
| --- | --- | --- |
| [HEX image](ek_ra8p1_vision_face_detection_yolo_fastest_TFLM_FSP660.hex) | 1,695,049 bytes | Select this image in Renesas Flash Programmer |

## Build information

| Item | Value |
| --- | --- |
| Devices Supported | EK-RA8P1 |
| e² studio / FSP | 2026-07 / 6.6.0 |
| Toolchain | LLVM/Clang (ATfE 22.1.0) |
| Build configuration | e² studio debug |
| Programming | Intel HEX + RFP; V3.23.00 used in the reported board tests |
| Source project (.zip) | [Download](../../ek_ra8p1_vision_face_detection_yolo_fastest_TFLM_FSP660.zip) |

## Memory allocation

| Item | Value |
| --- | --- |
| Model data | 369,968 bytes, internal MRAM |
| Tensor arena | 388,512 bytes, on-chip SRAM |
| Internal MRAM image span | About 589 KiB of 1 MiB; `0x02000000`–`0x020931C7` (with gaps) |
| External OSPI programming | Not required |

The HEX includes internal MRAM and startup option words in Config Areas 1/2. It contains no SRAM, SDRAM or OSPI address records.

## Results & Performance

| Stage | Time (ms) | Notes |
| --- | --- | --- |
| Model invocation time | ~4 | LCD inference time; NPU-accelerated, 108/108 operators on the NPU |

The LCD shows the latest model invocation time, excluding input copy, post-processing, camera capture and LCD refresh. It is not full pipeline latency.

## Program and run

Follow [Quick Deploy](../README.md#program-a-hex-image-with-rfp). Leave RFP external memory disabled and ensure Code Secure covers the image before programming.

After writing and verification, close RFP and reset or power-cycle the board. Confirm that the camera image and detected faces appear on the LCD.
