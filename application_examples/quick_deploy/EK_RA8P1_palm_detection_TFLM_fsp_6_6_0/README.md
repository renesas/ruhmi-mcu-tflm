# Palm Detection (TFLM) - EK-RA8P1

The camera feeds a single INT8 palm detector with a 192 x 192 input. The detector runs on every AI processing frame, and the LCD displays up to eight expanded hand bounding boxes with temporal filtering. This example does not include hand landmark inference or gesture classification.

## Files

| File | Size | Description |
| --- | --- | --- |
| [HEX image](ek_ra8p1_vision_palm_detection_model_camera_LCD_TFLM_FSP660.hex) | 4,295,804 bytes | Select this image in Renesas Flash Programmer |

## Build information

| Item | Value |
| --- | --- |
| Devices Supported | EK-RA8P1 |
| e² studio / FSP | 2026-07 / 6.6.0 |
| Toolchain | LLVM/Clang (ATfE 22.1.0) |
| Build configuration | e² studio debug |
| Runtime | TFLM + Ethos-U55; palm detection includes two CPU Gather operators |
| Programming | Intel HEX + Renesas Flash Programmer (RFP) |
| Source project (.zip) | - |

## Memory allocation

| Item | Value |
| --- | --- |
| Palm model data | 1,352,192 bytes, OSPI Flash load image copied to SDRAM |
| Tensor arena | 917,504 bytes (896 KiB), on-chip SRAM, 32-byte aligned |
| AI input image | 110,592 bytes, SDRAM |
| Internal MRAM image span | About 171 KiB of 1 MiB; `0x02000000`–`0x0202AC03` (with gaps) |
| OSPI model addresses | `0x90000000`–`0x9014A1FF` |

The HEX also includes startup option words in Config Areas 1/2. It contains no SRAM or SDRAM address records; startup copies the palm model from OSPI to SDRAM. The tensor arena size excludes other application buffers and stacks.

## Results & Performance

Successful programming of this HEX on the EK-RA8P1 using RFP has been confirmed.

| Stage | Time (ms) | Notes |
| --- | --- | --- |
| Model invocation time | ~38 | LCD `ai_inference_time_ms`; includes TFLM execution, CPU Gather operators and NPU hand-offs |

The LCD shows the latest model invocation time, excluding input copy, post-processing, camera capture and LCD refresh. It is not full pipeline latency or pure NPU execution time.

## Program and run

Follow [Quick Deploy](../README.md#program-a-hex-image-with-rfp). Enable **External Memory Settings**, select **EK-RA8P1**, and ensure the OSPI loader covers `0x90000000`–`0x9014A1FF`. Confirm **Code Secure = 1024 KB** before programming.

After successful writing and verification, close RFP and reset or power-cycle the board. Confirm the camera image and hand bounding boxes on the LCD. **SW2** pauses or resumes the pipeline.
