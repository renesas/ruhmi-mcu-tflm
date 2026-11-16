# RUHMI Framework TFLM Backend Scripts

This folder contains the scripts that compile and deploy AI models to Renesas RA8xx microcontrollers (MCU) and the Ethos-U55 NPU, using TensorFlow Lite for Microcontrollers (TFLM).

## Table of contents

1. [MCU Model Compiler](#1-mcu-model-compiler)
2. [Key Features](#2-key-features)
3. [Command-line Usage](#3-command-line-usage)
4. [YAML Config Mode](#4-yaml-config-mode)
5. [Deployment Examples](#5-deployment-examples)
6. [Calibration Data](#6-calibration-data)
7. [Generated Output Structure](#7-generated-output-structure)
8. [Helper Utilities](#8-helper-utilities)
9. [Compiling Multiple Models](#9-compiling-multiple-models)

---

## 1. MCU Model Compiler

`mcu_compile_tflm.py` compiles and deploys a `.tflite` model in one command: optional quantization (FP32 to INT8), NPU compilation with Arm Vela, and C/C++ code generation for the TFLM runtime. It can also run an optional host smoke test.

Run it from the **repository root folder**. Both scripts in this folder accept `--help` and `--version`.

## 2. Key Features

1. **TFLite input** — accepts `.tflite` models, FP32 or INT8.
2. **Quantization** — converts an FP32 model to INT8 using calibration data, with the `--quantize` option, and writes a quantization quality report.
3. **NPU support** — compiles the model for the Ethos-U55 with Arm Vela. Operators Vela cannot accelerate stay on the CPU.
4. **External memory support**:
   - **NPU**: `--external` makes Vela compile with the `Shared_Sram` memory mode, so weights can live in external memory.
   - **CPU**: `--external` only marks the output; you place the buffers in external memory yourself, using `__attribute__` sections.
   - **Auto-detection**: external memory mode turns on automatically for models larger than `--memory-threshold`.
5. **Host testing** — `--host-evaluate` runs the compiled CPU model once in the desktop TFLite interpreter as a smoke test.

## 3. Command-line Usage

```bash
python scripts/mcu_compile_tflm.py <model_path> <output_dir> [options]
```

#### Positional Arguments

| Argument | Description |
| :--- | :--- |
| `model_path` | Path to a `.tflite` model file, or a folder. A folder is searched recursively for `.tflite` files, and each one is compiled into its own output folder. |
| `output_dir` | Directory where the generated artifacts will be saved. |

#### Full Argument List

| Argument | Default | Description |
| :--- | :--- | :--- |
| **Target** | | |
| `--cpu` | `False` | Compile for the Cortex-M CPU (TFLM + CMSIS-NN). |
| `--npu` | `False` | Compile for the Ethos-U55 NPU (an INT8 model is required for acceleration). |
| **Precision** | | |
| `--quantize` | `False` | Quantize an FP32 model to INT8 before deployment. A model that is already quantized is not quantized again. |
| `--calib-data` | `""` | Path to calibration data: a `.npy` file, or a folder of `.npy` files. Requires `--quantize`. |
| `--calib-num` | `5` | Number of calibration samples to use. Requires `--quantize`. |
| **Memory & Optimization** | | |
| `--external` | `False` | Force external memory mode. NPU: Vela compiles with the `Shared_Sram` memory mode. CPU: only marks the output; you place the buffers yourself. |
| `--memory-threshold` | `0.8` | Model size (MB) above which external memory mode is turned on automatically. |
| `--memory-mode` | `Sram_Only` | (NPU only) Vela memory mode. Choices: `Sram_Only`, `Shared_Sram`. |
| `--optimization` | `Performance` | (NPU only) Vela optimization target. Choices: `Performance`, `Size`. |
| **Output & Naming** | | |
| `--suffix` | `""` | Adds a suffix to generated file and function names, so several models can live in one project (see [section 9](#9-compiling-multiple-models)). |
| `--result` | `""` | Path to write a JUnit XML result file. |
| **Host Testing** | | |
| `--host-evaluate` | `False` | Runs the compiled model once in the desktop TFLite interpreter with random input, as a smoke test. CPU only. |
| `--version` | | Print the compiler version. |

For NPU compilation, omitting `--optimization` (or `optimization` in YAML) selects `Performance` to prioritize execution speed. Use `--optimization Size` (or `optimization: Size` in YAML) to reduce runtime memory usage instead. This does not automatically place the tensor arena in SRAM.

Exactly one of `--cpu` or `--npu` is required. The compiler rejects options that do not apply to the selected target — for example `--memory-mode Shared_Sram` or `--optimization Size` together with `--cpu`, `--host-evaluate` together with `--npu`, or `--calib-data` without `--quantize`. A non-zero exit code means at least one model failed.

**Output folder naming:** `{model_name}_{TARGET}_tflm[_external][_quantized][_suffix]`

- **{model_name}** — taken from the input file, without its extension. Spaces are replaced with underscores (for example, `my model.tflite` becomes `my_model_CPU_tflm`).
- **{TARGET}** — `CPU` or `NPU`.
- **_tflm** — always added.
- **_external** — added when `--external` is given or the model is larger than `--memory-threshold`. With `--quantize`, an FP32 model is compared at a quarter of its size, its expected INT8 size.
- **_quantized** — added when `--quantize` actually converts an FP32 model. An already quantized model does not get it.
- **_suffix** — the `--suffix` value, if given (see [section 9](#9-compiling-multiple-models)).

## 4. YAML Config Mode

Instead of typing every option on the command line, you can put them in a YAML file and pass that file as the only argument:

```bash
python scripts/mcu_compile_tflm.py my_config.yaml
```

The script detects YAML mode automatically: if the first argument ends in `.yaml` or `.yml`, it is treated as a config file instead of a model path, and other command-line options are ignored.

**Example `my_config.yaml`:**

```yaml
model_path: models/model_fp32.tflite
output_dir: deploy_output
target: npu          # 'cpu' or 'npu' (required)
quantize: true
calib_data: calibration/
calib_num: 20
external: false
memory_threshold: 0.8
memory_mode: Sram_Only
optimization: Performance
suffix: ''
host_evaluate: false
result: ''
```

Each field maps to the matching command-line option above (for example, `memory_threshold` is the same as `--memory-threshold`). Only `model_path`, `output_dir`, and `target` are required; everything else falls back to its default value. Unknown keys are ignored with a warning. An optional `compiler: tflm` key is accepted; any other `compiler` value is an error. Relative paths are resolved from the folder you run the command in, not from the folder containing the YAML file.

This mode is useful for repeatable builds (for example, in CI), where you want to keep the exact compiler settings for a model in version control.

## 5. Deployment Examples

Replace the model, calibration, and output names in these examples with your own files and folders. Prepare the calibration data from real model inputs; see [Calibration Data](#6-calibration-data). To try the compiler first, use the [Quick Start](../README.md#quick-start) sample model.

On Windows PowerShell, put the command on one line: remove the trailing `\` and join the continued line.

**For NPU Deployment (pre-quantized INT8 model):**
```bash
python scripts/mcu_compile_tflm.py model_int8.tflite deploy_output --npu
```

**For NPU Deployment (FP32 model, quantized with calibration data):**
```bash
python scripts/mcu_compile_tflm.py model_fp32.tflite deploy_output --npu --quantize \
  --calib-data calibration_inputs.npy --calib-num 20
```

**For CPU Deployment:**
```bash
python scripts/mcu_compile_tflm.py model.tflite deploy_output --cpu
```

**For CPU Deployment (FP32 model, quantized):**
```bash
python scripts/mcu_compile_tflm.py model_fp32.tflite deploy_output --cpu --quantize \
  --calib-data calibration/ --calib-num 20
```

**For Large Models (External Memory):**
```bash
python scripts/mcu_compile_tflm.py large_model_int8.tflite deploy_output --npu --external
```

This flag selects how Vela lays out the model. The actual memory placement still depends on your project's source code and the linker configuration generated by FSP — see [Memory placement](../docs/tflm_backend/README.md#11-memory-placement).

**Every model in a folder, with a JUnit report:**
```bash
python scripts/mcu_compile_tflm.py models_int8 deploy_output --npu --result results.xml
```

**CPU host smoke test:**
```bash
python scripts/mcu_compile_tflm.py model.tflite deploy_output --cpu --host-evaluate
```

This runs the model once in the desktop TFLite interpreter with random input. It confirms that the model loads and runs, but it does not execute the generated embedded code and does not check accuracy.

## 6. Calibration Data

With `--quantize`, the compiler uses the TensorFlow Lite calibrator to produce a model with INT8 inputs, outputs, and activations, INT32 biases, and per-channel weights.

`--calib-data` accepts:

- a folder of `.npy` files, one sample per file, read in sorted file-name order (sub-folders are not searched). The first `--calib-num` files are used.
- a single `.npy` file. For a single-input model, the array is either one sample (the input shape, with or without its leading batch dimension of 1) or several samples stacked along a new first axis, of which the first `--calib-num` are used. Any other shape is an error. If the file holds fewer samples than `--calib-num`, the compiler warns and uses the ones available.

Each sample must match the model input's shape and value range, and must be preprocessed exactly like your real input. For a model with several inputs, each sample must be a list or tuple with one array per input, in input order, saved as a NumPy object array.

> [!CAUTION]
> For multi-input models, `.npy` files are loaded with `allow_pickle=True`. Only use calibration files from a source you trust. For single-input models, pickled files are rejected.

If `--calib-data` is not given, the compiler warns and uses random representative data. This is only meant for early bring-up and testing, and usually gives poor accuracy. A missing path, an empty folder, a non-`.npy` file, or a sample count below 1 stops the compilation with an error, instead of silently falling back to random data.

When the model is actually converted, the compiler compares the FP32 and INT8 models on the calibration samples and writes `quantization/quant_debug/model_stats.json`, with overall agreement metrics (cosine similarity, PSNR, MSE, maximum absolute error).

If the console warns about low cosine similarity (below 0.997) or PSNR (below 30 dB), review the calibration data before trusting the quantized model. These numbers measure agreement with the FP32 model, not task accuracy.

## 7. Generated Output Structure

The output directory contains the generated C/C++ code, plus a few other artifacts. Here is an example structure for a quantized NPU deployment:

```text
deploy_output/
└── model_NPU_tflm_quantized/
    ├── deploy/
    │   └── src/                       --> **Where the source code lives**
    │       ├── model.h
    │       ├── model.cc
    │       ├── model_data.h
    │       └── model_data.c
    ├── reference/
    │   └── model.tflite               # a copy of your original model
    ├── quantization/                  # only when --quantize converts the model
    │   ├── model_int8.tflite
    │   └── quant_debug/
    │       └── model_stats.json
    └── vela/                          # NPU builds only
        ├── model_int8_vela.tflite
        ├── model_int8_summary_RA8P1.csv
        └── vela_report.txt
```

The four files under **`deploy/src/`** are what you copy into an e² studio project. Compile `model.cc` as **C++17** and `model_data.c` as plain **C**. Keep the rest of the folder: the metrics utility reads it, and it is useful for debugging.

📖 See the [Runtime API guide](../docs/tflm_backend/README.md#10-runtime-api) to learn how to call the generated functions from your own application.

## 8. Helper Utilities

### Model Metrics Analysis (`scripts/utils/check_model_metrics_tflm.py`)

This helper script analyzes the memory usage (RAM/Flash), operation count (MACs), and NPU operator coverage of a compiled model.

```bash
python scripts/utils/check_model_metrics_tflm.py <path_to_model_output_dir>
```

See [scripts/utils/README.md](utils/README.md) for more details.

## 9. Compiling Multiple Models

If your application uses more than one model, compile each model separately, and give each one a different suffix with `--suffix`. This keeps the generated file and function names from colliding.

**Example:**

```bash
# Deploy the detector (NPU) with suffix "detector"
python scripts/mcu_compile_tflm.py detector.tflite deploy_output --npu --suffix detector

# Deploy the classifier (CPU) with suffix "classifier"
python scripts/mcu_compile_tflm.py classifier.tflite deploy_output --cpu --suffix classifier
```

The suffix is joined to every generated name with an underscore:

- Files: `model_detector.h`, `model_detector.cc`, `model_data_detector.h`, `model_data_detector.c`
- Functions: `ModelInit_detector()`, `RunModel_detector()`, `GetLastInvokeStatus_detector()`, ...
- Accessors: `GetModelInputPtr_detector_<tensor>()`, `GetModelOutputPtr_detector_<tensor>()`
- Helpers: `model_quantize_detector_input_<tensor>()`, `model_dequantize_detector_output_<tensor>()`, `model_quantize_detector()`, `model_dequantize_detector()`
- Macros: `DETECTOR_INPUT_SCALE_<TENSOR>`, `DETECTOR_OUTPUT_ZERO_POINT_<TENSOR>`, ...
- Output folder: `detector_NPU_tflm_detector`

Leading and trailing underscores are ignored, so `--suffix _detector` is the same as `--suffix detector`. Other characters that are not letters or digits become `_`. Because every name carries the suffix, the headers of several models can be included in the same source file.
