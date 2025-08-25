#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/workspace/QP-SNN-Quantization-pass"
VENV_DIR="/workspace/.venvs/qp-snn"
PIP_CACHE_DIR="/workspace/.cache/pip"
TMPDIR="/workspace/.tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
export PIP_CACHE_DIR TMPDIR TMP="$TMPDIR" TEMP="$TMPDIR"

# 1) 激活 venv
if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "[ERROR] 未找到 venv: $VENV_DIR，请先运行 scripts/setup_venv.sh" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install -q --upgrade pip setuptools wheel

# 2) 保护现有 GPU 版 PyTorch（已按 cu121 安装），不覆盖
python - <<'PY'
import torch
print(f"[INFO] Torch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
PY

# 3) 安装 MASE 本体为 editable（不自动拉取依赖）
python -m pip install -q --no-deps -e "$REPO_DIR/mase"

# 4) 一次性安装 MASE 依赖（基于 mase/setup.py，去除 torch/torchvision 以避免覆盖 GPU 轮子；去除 mase-triton/pycuda/tensorrt）
python -m pip install -q --upgrade-strategy only-if-needed \
  onnx black toml GitPython colorlog cocotb==1.9.2 pytest \
  pytorch-lightning transformers==4.51 timm pytorch-nlp datasets==3.3.2 \
  evaluate==0.4.3 ipython ipdb sentencepiece einops pybind11 tabulate tensorboardx \
  "hyperopt @ git+https://github.com/hyperopt/hyperopt.git" accelerate optuna \
  "stable-baselines3[extra]" h5py scikit-learn scipy==1.14 onnxruntime matplotlib \
  sphinx-rtd-theme "sphinx-needs>=4" "sphinx-test-reports @ git+https://github.com/useblocks/sphinx-test-reports" \
  sphinxcontrib-plantuml imageio imageio-ffmpeg opencv-python kornia ghp-import optimum \
  pytest-profiling myst_parser pytest-cov pytest-xdist pytest-sugar pytest-html lightning wandb \
  bitarray "bitstring>=4.2" emoji numpy==2.2.4 tensorboard sphinx_needs onnxconverter-common \
  absl-py sphinx-glpi-theme prettytable pyyaml pynvml cvxpy py-cpuinfo pandas psutil \
  Pillow==10.4.0 mpmath==1.3.0 myst-nb sphinx-book-theme pydot attr-dot-dict ultralytics

# 5) 关键导入自检
python - <<'PY'
mods = [
    ("chop", "chop"), ("toml", "toml"), ("tabulate", "tabulate"), ("evaluate", "evaluate"),
    ("transformers", "transformers"), ("datasets", "datasets"), ("numpy", "numpy"),
    ("scipy", "scipy"), ("tensorboard", "tensorboard"), ("tensorboardx", "tensorboardX"),
]
import importlib
failed=[]
for n,m in mods:
    try:
        importlib.import_module(m)
        print(f"[OK] import {n}")
    except Exception as e:
        print(f"[FAIL] import {n}: {e}")
        failed.append(n)
print("ALL_OK" if not failed else f"SOME_IMPORTS_FAILED: {failed}")
PY

echo "[INFO] MASE 安装完成（editable），依赖已批量安装。"
