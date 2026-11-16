
# License Summary

The RUHMI Framework AI MCU Compiler — TFLM backend is licensed per component, as listed below. Always defer to the SPDX license identifier or header comment in each individual source file.

## RUHMI FRAMEWORK AI MCU COMPILER — TFLM Backend

|Component|License|
|---|---|
|TFLM Backend Compiler (`scripts/mcu_compile_tflm.py`)|Apache License, Version 2.0|
|TFLM Backend Utility (`scripts/utils/check_model_metrics_tflm.py`)|BSD 3-Clause License|
|Generated wrapper source code (`model.h`, `model.cc`, `model_data.h`, and the non-data portions of `model_data.c`)|BSD 3-Clause License|

> **Notes**
>
> **The `model_data.c` byte array (the serialized/quantized/Vela-transformed model itself) retains the original customer model's license; the TFLM compiler does not relicense customer model weights.**
>
> **TFLM, CMSIS-NN, and Ethos-U runtime components are supplied separately via FSP under their own respective licenses; please refer to the copyright and license notice contained in each source-code file.**

## Embedded Applications

For embedded applications, please refer to the copyright and license notices in the source files and the license files included with each project.

## External Licenses

In the procedure to install it into your host machine, some software components from the external place will be downloaded and installed automatically. Please refer to the license information for each component listed below.

|Component|License|
|---|---|
|tensorflow/tensorflow|Apache License, Version 2.0|
|numpy/numpy|BSD 3-Clause License|
|Arm Ethos-U Vela (`ethos-u-vela`)|Apache License, Version 2.0|
|yaml/pyyaml|MIT|
|weiwei/junitparser|Apache License, Version 2.0|
