# Check Model Metrics (`check_model_metrics_tflm.py`)

This script analyzes the memory requirements (RAM and Flash) and operations (MACs) of a model compiled with `scripts/mcu_compile_tflm.py`. It works for both CPU and NPU builds, and needs only the Python standard library.

## Usage

Run it from the repository root:

```bash
python scripts/utils/check_model_metrics_tflm.py <path_to_model_output_dir>
```

**Argument:**
`<path_to_model_output_dir>`: A single model's output folder, the one that contains `deploy/` (for example `deploy_output/ad01_int8_NPU_tflm`). Its `deploy/src/` folder is also accepted. If you pass the parent folder by mistake, the error lists the model output folders it contains.

**Options:**

| Option | Description |
| :--- | :--- |
| `--suffix NAME` | Select one model when several models' sources were copied into one `deploy/src/`. Use the `--suffix` the model was compiled with. |
| `--json` | Print the metrics as JSON instead of a table. The exit code is non-zero when the folder cannot be analyzed. |
| `--version` | Print the script version. |

The script reads:

- `deploy/src/model*.cc` — tensor arena size, registered operators, number of inputs and outputs
- `deploy/src/model_data*.h` — model data size
- `quantization/` or `reference/` — the `.tflite` model used to count CPU MACs
- `vela/vela_report.txt` — MAC count and NPU/CPU operator split (NPU builds only)

## Metrics Explanation

| Metric | Description | Note |
| :--- | :--- | :--- |
| **RAM (Tensor Arena)** | `TENSOR_ARENA_SIZE` reserved in `model.cc`. | An estimate from the compiler, not a measurement. Check `GetTensorArenaUsedBytes()` on the board. If Vela leaves operators on the CPU, the report adds a note because the arena is then a host-side estimate. |
| **Flash (Model/Weights)** | Size of the model byte array in `model_data.c`. | Does not include program code. |
| **MAC Operations** | Total multiply-accumulate operations. | NPU: taken from the Vela report. CPU: computed for convolution (2D and 3D), depthwise convolution, transpose convolution, fully connected, batch matrix multiplication, and unidirectional sequence LSTM layers, including batch and sequence dimensions; other operators count as zero. |
| **NPU Acceleration** | Share of operators Vela placed on the NPU. | `(NPU operators / total operators) × 100`. Not a latency measure. |

## Example Output

For the [Quick Start](../../README.md#quick-start) sample:

```text
Memory Analysis for: deploy_output/ad01_int8_NPU_tflm
--------------------------------------------------
Platform             : NPU
Accelerator          : Ethos_U55_256  Config: RA8P1  Mode: Sram_Only

RAM  (Tensor Arena)  :        1,568 bytes  (1.5 KB)
Flash (Model/Weights):      203,136 bytes  (0.19 MB)
MAC Operations       :      264,192 ops    (0.26 M)
NPU Acceleration     :       100.0%      (10 NPU / 10 total ops, 0 CPU fallback)

Inputs / Outputs     : 1 input(s), 1 output(s)
Registered Ops ( 1)  : AddCustom
TFLite source        : ad01_int8.tflite
Vela report          : deploy_output/ad01_int8_NPU_tflm/vela/vela_report.txt
--------------------------------------------------
```

> **Note:** CPU and NPU MAC counts are calculated differently, so do not compare them directly. Confirm memory use and performance on real hardware with the [benchmark project](../../docs/benchmark/README.md).

## Python API

`get_metrics()` returns the same values without printing, for use in your own scripts. Run from the repository root:

```python
import sys
from pathlib import Path

sys.path.insert(0, "scripts/utils")
from check_model_metrics_tflm import get_metrics

metrics = get_metrics(Path("deploy_output/ad01_int8_NPU_tflm"))
if metrics["error"] is None:
    print(metrics["platform"], metrics["arena_bytes"], metrics["mac_count"])
```

The returned dictionary always contains `platform`, `arena_bytes`, `flash_bytes`, `mac_count`, `npu_mac_count`, `npu_util_pct`, and `error`. Always check `error` first. On success it also contains `vela_sram_bytes`, `vela_flash_bytes`, `npu_ops`, `cpu_ops`, `num_inputs`, `num_outputs`, `registered_ops`, and `warnings`; the `--json` output has the same fields. `npu_mac_count` is an approximation: total MACs multiplied by the NPU operator percentage. Pass `suffix="name"` to select one model in a shared `deploy/src/`.
