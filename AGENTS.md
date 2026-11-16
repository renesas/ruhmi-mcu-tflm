# Agent Guide for Users of This Repository

For AI agents helping a user **use** the RUHMI TFLM backend: compile a `.tflite` model to C/C++ for Renesas RA8xx MCUs (Cortex-M CPU or RA8P1 Ethos-U55 NPU), integrate it into an e² studio/FSP project, or run the prebuilt demos. This page answers most tasks directly; open a linked guide only when you need more detail.

Host results (compile success, metrics, `--host-evaluate`) do not prove on-board correctness, accuracy, memory use, or speed.

<a id="1-reading-the-repository-efficiently"></a>

## 1. Reading the repository efficiently

These are defaults to save tokens. If the user explicitly asks to inspect a file, do so.

**Data files — skip and exclude from searches** (for example `rg --glob '!**/model_data*.c'`). They are generated or binary and contain no logic:

| Pattern | Contents |
| --- | --- |
| `**/model_data*.c` | Model byte arrays, 2–8 MB each. Read `model.h` instead for tensor names, types, and scales. |
| `**/bg_font_*.c` | Font bitmaps |
| `**/*.hex`, `**/*.zip`, `**/*.tflite`, `**/*.npy` | Firmware, archives, models, calibration data |
| `docs/assets/**`, `application_examples/**/docs/**` | Images and GIFs |
| `tflm-env/`, compiler output folders (except `deploy/src/model.h`, `model.cc`, `vela/vela_report.txt`) | Virtual environment and generated artifacts |

**Large source — read the relevant part, not the whole file.** For ordinary compile tasks, §4 and `--help` are enough. When the user wants to understand, debug, or change the compiler, `scripts/mcu_compile_tflm.py` (~2,500 lines) is in scope. Search for the function you need and read only that one:

| Topic | Functions in `scripts/mcu_compile_tflm.py` |
| --- | --- |
| CLI, YAML, option validation | `create_parser`, `load_config`, `_validate_tflm_options`, `main` |
| Per-model pipeline (start here) | `compile_model_tflm` |
| Model parsing, operator mapping | `parse_tflite_model`, `TFLITE_BUILTIN_OP_MAP`, `TFLM_CUSTOM_OP_MAP` |
| Quantization and calibration | `quantize_tflite_model`, `_load_tflm_calibration_samples`, `run_tflite_quantization_quality_check` |
| Vela / NPU | `run_vela`, `_vela_launcher` |
| Arena size and external memory decisions | `_estimate_arena_size`, `_tflm_npu_arena_overhead`, `needs_external_memory` |
| C/C++ code generation | `_generate_model_files` (`model.h`/`model.cc`), `_generate_model_data_files` |
| Host smoke test, JUnit | `_tflm_host_evaluate`, `write_junit_xml` |

The metrics tool `scripts/utils/check_model_metrics_tflm.py` (~650 lines) is small enough to read when needed.

<a id="2-task-router"></a>

## 2. Task router

| User wants to… | Do this | Detail, only if needed |
| --- | --- | --- |
| Set up the host tools | [§3](#3-environment) | [install/README.md](install/README.md) |
| Compile a model (CPU or NPU) | [§4](#4-compile-a-model) | [scripts/README.md](scripts/README.md) |
| Check size, MACs, NPU coverage | [§4](#4-compile-a-model), last command | [scripts/utils/README.md](scripts/utils/README.md) |
| Call the generated code from an app | [§5](#5-use-the-generated-code) | [Runtime API](docs/tflm_backend/README.md#10-runtime-api) |
| Fit a large model / fix flash errors | [§6](#6-memory-and-flash) | [Memory placement](docs/tflm_backend/README.md#11-memory-placement) |
| Measure latency on EK-RA8P1 | Import `docs/benchmark/Benchmark_TFLM_v660.zip`, copy the four files into `src/ruhmi_perf_eval/ruhmi_inference_code/` | [docs/benchmark/README.md](docs/benchmark/README.md) |
| Run a ready demo without building | Program a prebuilt HEX with Renesas Flash Programmer | [quick_deploy/README.md](application_examples/quick_deploy/README.md) |
| Modify a demo application | Import its project ZIP from `application_examples/` | [application_examples/README.md](application_examples/README.md) |
| Diagnose an error | [§7](#7-common-problems) | [Troubleshooting](docs/tflm_backend/README.md#13-troubleshooting), [known issues](docs/known_issues/README.md) |
| Understand or change compiler behavior | [§1](#1-reading-the-repository-efficiently) function map | `scripts/mcu_compile_tflm.py` |

<a id="3-environment"></a>

## 3. Environment

Host: Ubuntu 22.04/24.04 or Windows 10/11, **Python 3.10 only**. Pinned: TensorFlow 2.18.0, numpy 1.26.4, ethos-u-vela 5.2.0. Target: e² studio with **FSP 6.6.0** (the only validated version). Run all commands from the repository root.

```bash
python3.10 -m venv tflm-env && source tflm-env/bin/activate     # Windows: py -3.10 -m venv tflm-env; .\tflm-env\Scripts\Activate.ps1
pip install --upgrade pip && pip install -r requirements-tflm.txt
python scripts/mcu_compile_tflm.py --version                    # expect 1.0.0
python -m ethosu.vela --version                                  # expect 5.2.0
```

If an environment already exists, activate it instead of reinstalling. Ask before installing system packages.

<a id="4-compile-a-model"></a>

## 4. Compile a model

```bash
# NPU, model already INT8
python scripts/mcu_compile_tflm.py model_int8.tflite deploy_output --npu
# NPU, FP32 model quantized with real calibration data (.npy file or folder of .npy)
python scripts/mcu_compile_tflm.py model_fp32.tflite deploy_output --npu --quantize --calib-data calib/ --calib-num 20
# CPU (FP32 or INT8)
python scripts/mcu_compile_tflm.py model.tflite deploy_output --cpu
# Several models in one app: give each a unique suffix
python scripts/mcu_compile_tflm.py detector.tflite deploy_output --npu --suffix detector
# Every .tflite in a folder, with a JUnit report
python scripts/mcu_compile_tflm.py models/ deploy_output --npu --result results.xml
# Metrics: pass ONE model's output folder (the one containing deploy/), not deploy_output/
python scripts/utils/check_model_metrics_tflm.py deploy_output/<model>_NPU_tflm
```

Rules:

- Exactly one of `--cpu` / `--npu`. Only `.tflite` input. NPU acceleration needs an INT8 model; unsupported operators fall back to the CPU.
- NPU-only options: `--memory-mode {Sram_Only,Shared_Sram}`, `--optimization {Performance,Size}` (default `Performance`). CPU-only: `--host-evaluate` (random-input smoke test, not accuracy).
- `--calib-data` requires `--quantize`. Calibration samples must match the real preprocessing, shape, range, and input order. Without `--calib-data` the compiler uses random data — bring-up only, poor accuracy; tell the user.
- For a single-input model, a single `.npy` file must hold one sample in the input shape or several samples stacked along a new first axis; other shapes and pickled files are rejected.
- After quantization, low cosine similarity (< 0.997) or PSNR (< 30 dB) warnings mean: check `quantization/quant_debug/model_stats.json` and improve calibration data.
- YAML mode: `python scripts/mcu_compile_tflm.py config.yaml` (keys mirror the options with underscores; `model_path`, `output_dir`, `target` required; unknown keys are ignored with a warning). See [YAML Config Mode](scripts/README.md#4-yaml-config-mode).
- Non-zero exit code means at least one model failed.

Output folder: `deploy_output/{model}_{CPU|NPU}_tflm[_external][_quantized][_suffix]/` (`_quantized` only when an FP32 model is actually converted)

```text
deploy/src/   model.h  model.cc  model_data.h  model_data.c   <- copy these four into the project
reference/    original model
quantization/ model_int8.tflite, quant_debug/model_stats.json   (with --quantize)
vela/         vela_report.txt (NPU/CPU operator split), *_summary_RA8P1.csv       (NPU only)
```

<a id="5-use-the-generated-code"></a>

## 5. Use the generated code

FSP project: enable Google TFLM Core Lib + CMSIS-NN Kernel; for NPU also the TFLM Ethos-U Kernel and Ethos-U driver. Compile `model.cc` as **C++17** (`-fno-rtti -fno-exceptions`) and `model_data.c` as **C**. Defines: `TF_LITE_STATIC_MEMORY`, `FLATBUFFERS_LOCALE_INDEPENDENT=0`, `CMSIS_NN`, `KERNELS_OPTIMIZED_FOR_SPEED`, plus `ETHOSU55` for NPU.

When `--suffix detector` is used:

- Lifecycle/status functions gain a trailing suffix: `ModelInit_detector()`, `RunModel_detector()`, `GetLastInvokeStatus_detector()`.
- Tensor accessors put the suffix before the tensor identifier: `GetModelInputPtr_detector_<tensor>()`, `GetModelOutputPtr_detector_<tensor>()`.
- Quantization macros take the suffix as an upper-case prefix: `DETECTOR_INPUT_SCALE_<TENSOR>`, `DETECTOR_OUTPUT_ZERO_POINT_<TENSOR>`.

Always use the actual names and types declared in the generated header; do not infer every name by appending a suffix.

Illustrative startup and inference sequence for **INT8 quantized input and output**, without a suffix, inside a `void` application function. Replace `<tensor>` with the respective input/output identifier from the generated header, and supply valid `x` and `i`; this is not directly copyable code. For FP32 I/O, use the generated float pointers and read/write float values directly instead of using INT8 quantization helpers. Report failures through the application's error handling and stop before accessing invalid input or output buffers.

```c
/* After board, Ethos-U driver (NPU), and external memory are initialized */
if (ModelInit() != 0) {
    /* -1: bad model data, -2: arena too small / missing op or FSP kernel */
    return;
}
int8_t *in = GetModelInputPtr_<tensor>();      /* type matches the tensor; size is NOT in model.h */
in[i] = model_quantize_input_<tensor>(x);       /* helpers + SCALE/ZERO_POINT macros for quantized I/O */
RunModel(false);
if (GetLastInvokeStatus() != 0) {
    /* Inference failed; do not read the output. */
    return;
}
float y = model_dequantize_output_<tensor>(GetModelOutputPtr_<tensor>()[i]);
/* Repeat from GetModelInputPtr for each inference; never call ModelInit() again. One thread at a time. */
```

Tensor element counts come from the model shapes printed by the compiler; never write past a buffer. `TENSOR_ARENA_SIZE` in `model.cc` is an estimate — compare `GetTensorArenaUsedBytes()` with `GetTensorArenaSizeBytes()` on the board, and increase it if `ModelInit()` returns `-2`. Unknown custom ops leave a `TODO: unregistered custom op` in `model.cc` that the user must implement. Standard operators that the FSP TFLM library cannot run (for example `LSTM`, `RNN`, `SELECT`) stop compilation with an error instead.

<a id="6-memory-and-flash"></a>

## 6. Memory and flash

- Model data kept in internal flash shares that space with the program. If the image does not fit, move the model data to OSPI, and confirm the final layout in the linker map file.
- Move model data to OSPI: in `model_data.c`, `__attribute__((aligned(16), section(".ospi0_cs1")))`. Move a large arena to SDRAM: in `model.cc`, `__attribute__((aligned(16), section(".sdram")))`. Verify section names against the project's FSP linker script.
- The tensor arena must be **writable RAM** — never flash or OSPI. Initialize OSPI/SDRAM before `ModelInit()`. Keep 16-byte alignment and existing cache maintenance at CPU/NPU hand-offs.
- `--external` (or auto-detection above 0.8 MB, `--memory-threshold`; with `--quantize`, an FP32 model is compared at a quarter of its size) makes Vela use `Shared_Sram` for NPU builds; for CPU builds it only marks the output. It never moves model data by itself.

<a id="7-common-problems"></a>

## 7. Common problems

| Symptom | Action |
| --- | --- |
| `Vela not found` | Install requirements into the active env; check `python -m ethosu.vela --version` |
| `Scale and zero points should not be recorded before quantization` | Model is partially quantized: drop `--quantize` or start from a clean FP32 model |
| Many CPU-fallback operators | Read `vela/vela_report.txt`; quantize FP32 models first |
| `ModelInit()` returns `-2` | Arena size, missing FSP kernel components, or unregistered custom op |
| `Model uses builtin operator(s) that the FSP TFLM library cannot run` | Replace or remove the listed operators in the source model |
| Missing `ethosu.h` / `Register_ETHOSU` | Enable TFLM Ethos-U kernel and Ethos-U driver in FSP |
| Hard fault reading model or allocating tensors | Check sections, external-memory init order, writability, alignment, cache |
| Debug fails with `Cannot find smart bundle` | Re-import via **File > Import > Rename & Import Existing C/C++ Project Into Workspace** |
| INT8 Transpose of rank ≥ 5 executed on CPU, including CPU fallback in NPU applications | Check the TFLM CMSIS-NN adapter actually bundled in the project. In the affected adapter, rank 5 may be silently skipped while reporting success; rank > 5 is rejected during preparation. See [known issue](docs/known_issues/cmsis_nn_transpose_rank_limit.md). |

<a id="8-demos-and-board-safety"></a>

## 8. Demos and board safety

- Five prebuilt HEX demos exist in `application_examples/quick_deploy/`, but source project ZIPs exist only for **face detection, image classification, and palm + landmark + gesture**. Palm-only and palm + landmark are HEX-only.
- Loose `application_examples/*/src/` folders are for reading; build from the imported ZIP project. Editing source does not change any prebuilt HEX.
- **Never flash, erase, or change flash options/security boundaries without the user's explicit approval.** Follow [quick_deploy/README.md](application_examples/quick_deploy/README.md) exactly; never use chip erase, OTP, lockbit, or device initialization.

<a id="9-reporting-to-the-user"></a>

## 9. Reporting to the user

State what was run on the host and what still needs to be verified on the board. Present arena size, MACs, and NPU coverage as estimates, not latency or accuracy. Say clearly when random calibration data was used. Write outputs to a new folder; do not overwrite the user's existing deployments.
