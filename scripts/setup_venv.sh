#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/workspace/QP-SNN-Quantization-pass"
VENV_DIR="/workspace/.venvs/qp-snn"
PIP_CACHE_DIR="/workspace/.cache/pip"
TMPDIR="/workspace/.tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
export PIP_CACHE_DIR
export TMPDIR
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] 未找到 python3" >&2
  exit 1
fi

# 如果 venv 目录不存在或缺少激活脚本，则创建 venv
if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "[INFO] 创建 venv: $VENV_DIR"
  rm -rf "$VENV_DIR" 2>/dev/null || true
  python3 -m venv "$VENV_DIR"
fi

# 激活 venv
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install -q --upgrade pip setuptools wheel

# 选择 PyTorch CUDA 版本（默认 cu121，适配 H100/H200）
CUDA_VERSION="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\).*/\1/p' | head -n1 || true)"
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
if [ -z "${CUDA_VERSION:-}" ]; then
  echo "[WARN] 未检测到 nvidia-smi 或 CUDA 版本，按 cu121 安装 PyTorch"
else
  echo "[INFO] 检测到 CUDA 版本: ${CUDA_VERSION}（驱动） -> 使用 cu121 轮子"
fi

# 安装 PyTorch 三件套（使用持久缓存与大临时目录）
python -m pip install -q --index-url "$PYTORCH_INDEX_URL" torch torchvision torchaudio

# 安装其余依赖
python -m pip install -q -r "$REPO_DIR/requirements.txt"

# 简单导入校验
python - <<'PY'
import importlib
mods = [
    ("torch", "torch"),
    ("timm", "timm"),
    ("spikingjelly", "spikingjelly"),
    ("einops", "einops"),
    ("PIL", "PIL.Image"),
    ("yaml", "yaml"),
    ("cv2", "cv2"),
    ("pandas", "pandas"),
    ("sklearn", "sklearn"),
    ("scipy", "scipy"),
    ("tqdm", "tqdm")
]
failed = []
for name, module in mods:
    try:
        importlib.import_module(module)
        print(f"[OK] import {name}")
    except Exception as e:
        print(f"[FAIL] import {name}: {e}")
        failed.append(name)
print("ALL_OK" if not failed else f"SOME_IMPORTS_FAILED: {failed}")
PY

# spikingjelly 兜底：若导入失败，则从 Git 安装最新版本
if ! python - <<'PY'
import importlib, sys
importlib.import_module("spikingjelly")
PY
then
  echo "[INFO] 尝试从 Git 安装 spikingjelly"
  python -m pip install -q "git+https://github.com/fangwei123456/spikingjelly"
fi

echo "[INFO] venv 就绪: $VENV_DIR"
echo "[HINT] 使用命令: source $REPO_DIR/scripts/activate_venv.sh"
