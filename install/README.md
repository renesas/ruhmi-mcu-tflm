# Installation

This guide installs the host tools of the RUHMI Framework TFLM backend: the compiler `scripts/mcu_compile_tflm.py`, the metrics utility `scripts/utils/check_model_metrics_tflm.py`, and their Python dependencies.

Select your installation guide:
- [Installation guide for Ubuntu Linux](#installation---ubuntu-linux)
- [Installation guide for Windows](#installation---windows)

## Installation - Ubuntu Linux

You will need:

- A machine with Ubuntu 22.04 or Ubuntu 24.04 (Ubuntu 22.04 is recommended)
- Python 3.10.x, through the system Python, PyEnv, or another Python version manager

**Prepare the environment**

Ubuntu 22.04 ships Python 3.10 by default:

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip git
```

On Ubuntu 24.04, install Python 3.10 from the deadsnakes PPA (or use PyEnv):

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-venv
```

**Install the TFLM backend**

```bash
git clone https://github.com/renesas/ruhmi-mcu-tflm.git
cd ruhmi-mcu-tflm

python3.10 -m venv tflm-env
source tflm-env/bin/activate

pip install --upgrade pip
pip install -r requirements-tflm.txt
```

Your prompt should now show that you are inside the virtual environment:
`(tflm-env) user@compute:~/ruhmi-mcu-tflm$`

Continue with [Verify the installation](#verify-the-installation).

## Installation - Windows

TensorFlow on Windows requires the Microsoft Visual C++ Redistributable. Many machines already have it, because other software installs it too. If `import tensorflow` fails with a missing `msvcp140.dll` or `msvcp140_1.dll` error, download and install [this package](https://aka.ms/vs/17/release/vc_redist.x64.exe).

Install Python 3.10 from [Python 3.10.11](https://www.python.org/downloads/release/python-31011/), the last 3.10 release with a Windows installer.

Open **PowerShell** from the Windows start menu, then clone the repository and move into it:

```powershell
git clone https://github.com/renesas/ruhmi-mcu-tflm.git
cd ruhmi-mcu-tflm
```

**Prepare the virtual environment**

```powershell
py -3.10 -m venv tflm-env
```

Before activating the virtual environment, you may need to change the execution policy for the current PowerShell session:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\tflm-env\Scripts\Activate.ps1
```

You will see the prompt as `(tflm-env) PS C:\...\ruhmi-mcu-tflm>`.

> [!TIP]
> If your organization does not allow script activation, skip `Activate.ps1` and call `.\tflm-env\Scripts\python.exe` wherever this guide uses `python`. The compiler still finds Arm Vela without an activated environment.

**Install the TFLM backend**

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-tflm.txt
```

## Verify the installation

Run these commands from the repository root. Each one should print the version shown below it.

```
python scripts/mcu_compile_tflm.py --version
mcu_compile_tflm.py 1.0.0

python scripts/utils/check_model_metrics_tflm.py --version
check_model_metrics_tflm.py 1.0.0

python -m ethosu.vela --version
5.2.0

python -c "import tensorflow as tf; print(tf.__version__)"
2.18.0
```

You are now ready to compile a model. Continue with the [Quick Start](../README.md#quick-start) or the [TFLM Backend Guide](../docs/tflm_backend/README.md).

## Python dependencies

All dependencies are listed in [`requirements-tflm.txt`](../requirements-tflm.txt):

| Package | Version | Used for |
|---------|---------|----------|
| `tensorflow` | 2.18.0 | Quantization, quantization-quality report, host evaluation, arena estimation |
| `numpy` | 1.26.4 | Arrays and calibration data |
| `ethos-u-vela` | 5.2.0 | NPU compilation (`--npu`) |
| `PyYAML` | 6.x | YAML configuration mode |
| `junitparser` | 3.x | JUnit XML output (`--result`) |

Arm Vela is only needed for NPU builds, and the metrics utility uses only the Python standard library.

## Target tools

To build and run the generated code on a board, install **e² studio** with **Renesas FSP**. The [benchmark project](../docs/benchmark/README.md) targets FSP 6.6.0 and the LLVM toolchain. The required FSP components are listed in [FSP project requirements](../docs/tflm_backend/README.md#9-fsp-project-requirements).

> [!NOTE]
> The TFLM backend has been tested and validated with **Renesas FSP 6.6.0**, using the TFLM, CMSIS-NN, and Arm Ethos-U stacks bundled with FSP. Other FSP versions may work but have not been validated.
