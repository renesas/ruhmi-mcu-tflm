# Image Classification (MobileNet V1, TFLM) - EK-RA8P1

Prebuilt MobileNet V1 INT8 image classification example. The camera feeds a 224 x 224 input to TFLM on the Ethos-U55 NPU; the LCD shows smoothed Top-5 ImageNet results.

## Files

| File | Size | Description |
| --- | --- | --- |
| [HEX image](ek_ra8p1_vision_image_classification_mobilenet_v1_TFLM_FSP660.hex) | 2,000,318 bytes | Select this image in Renesas Flash Programmer |

## Build information

| Item | Value |
| --- | --- |
| Devices Supported | EK-RA8P1 |
| e² studio / FSP | 2026-07 / 6.6.0 |
| Toolchain | LLVM/Clang (ATfE 22.1.0) |
| Build configuration | e² studio debug |
| Programming | Intel HEX + RFP; V3.23.00 used in the reported board tests |
| Source project (.zip) | [Download](../../ek_ra8p1_vision_image_classification_mobilenet_v1_TFLM_FSP660.zip) |

## Memory allocation

| Item | Value |
| --- | --- |
| Model data | 473,744 bytes, OSPI Flash load image copied to SDRAM |
| Tensor arena | 422,688 bytes, on-chip SRAM |
| RGB888 input buffer | 150,528 bytes, on-chip SRAM |
| Internal MRAM image span | About 232 KiB of 1 MiB; `0x02000000`–`0x02039F5F` (with gaps) |
| OSPI model addresses | `0x90000000` to `0x90073A8F` |

The HEX also includes startup option words in Config Areas 1/2. It contains no SRAM or SDRAM address records; startup copies the model from OSPI to SDRAM.

## Results & Performance

| Stage | Time (ms) | Notes |
| --- | --- | --- |
| Model invocation time | ~2 | LCD inference time; NPU-accelerated, 57/57 operators on the NPU |

The LCD shows the latest model invocation time, excluding input copy, post-processing, camera capture and LCD refresh. It is not full pipeline latency.

## Program and run

Follow [Quick Deploy](../README.md#program-a-hex-image-with-rfp). Enable **External Memory Settings**, select **EK-RA8P1**, and ensure the OSPI loader covers `0x90000000`–`0x90073A8F`. Confirm **Code Secure = 1024 KB** before programming.

After successful writing and verification, close RFP and reset or power-cycle the board. Confirm the camera image and Top-5 labels on the LCD.
