#!/usr/bin/env bash
VENV_DIR="/workspace/.venvs/qp-snn"
if [ ! -d "$VENV_DIR" ]; then
  echo "[ERROR] 未找到虚拟环境：$VENV_DIR。请先运行 scripts/setup_venv.sh" >&2
  return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
export PYTHONPATH="/workspace/QP-SNN-Quantization-pass:/workspace/QP-SNN-Quantization-pass/mase/src:/workspace/QP-SNN-Quantization-pass/Spikingformer/dvs128-gesture:${PYTHONPATH:-}"
python -V
which python
echo "[INFO] venv 已激活，并设置 PYTHONPATH 指向仓库根与 mase/src"

# --- ensure tmux is installed (persistent) ---
if ! command -v tmux >/dev/null 2>&1; then
  echo "[INFO] 未检测到 tmux，尝试安装..."
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
      sudo apt-get update -y -qq && sudo apt-get install -y -qq tmux || echo "[WARN] tmux 安装失败，请手动安装。"
    else
      apt-get update -y -qq && apt-get install -y -qq tmux || echo "[WARN] tmux 安装失败，请手动安装。"
    fi
  else
    echo "[WARN] 未找到 apt-get，无法自动安装 tmux。请手动安装。"
  fi
fi
if command -v tmux >/dev/null 2>&1; then
  echo "[INFO] tmux 已就绪: $(tmux -V)"
else
  echo "[WARN] tmux 仍不可用。"
fi
