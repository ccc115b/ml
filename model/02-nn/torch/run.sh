#!/usr/bin/env bash
# 依序執行測試與所有 PyTorch 範例；任何一個失敗就停止。
# 用法：./run.sh                  執行全部（先跑 tests）
#       ./run.sh tests            只跑測試
#       ./run.sh main-gpt         只執行指定範例
# 裝置：自動選擇 CUDA / Apple MPS / CPU；可用 DEVICE=cpu|mps|cuda 指定，例如 DEVICE=cpu ./run.sh
# 範例：main-mlp main-cnn main-gru main-gpt main-pretrain main-finetune main-diffusion
#       main-pretrain 產生 pretrained.pt，main-finetune 需要它，所以要先跑 main-pretrain
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
examples=("$@")
[ ${#examples[@]} -eq 0 ] && examples=(tests main-mlp main-cnn main-gru main-gpt main-pretrain main-finetune main-diffusion)

for name in "${examples[@]}"; do
    if [ ! -f "$name.py" ]; then
        echo "找不到範例：$name.py" >&2
        exit 1
    fi
    echo "=================== $name ==================="
    "$PYTHON" "$name.py"
    echo
done
echo "全部完成。"
