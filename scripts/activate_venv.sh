#!/usr/bin/env bash
VENV_DIR="/workspace/.venvs/qp-snn"
if [ ! -d "$VENV_DIR" ]; then
  echo "[ERROR] 未找到虚拟环境：$VENV_DIR。请先运行 scripts/setup_venv.sh" >&2
  return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
export PYTHONPATH="/workspace/QP-SNN-Quantization-pass:/workspace/QP-SNN-Quantization-pass/mase/src:${PYTHONPATH:-}"
python -V
which python
echo "[INFO] venv 已激活，并设置 PYTHONPATH 指向仓库根与 mase/src"
