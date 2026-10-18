#!/usr/bin/env bash
# 依序執行測試與所有範例；任何一個失敗就停止。
# 用法：./run.sh                    執行全部（先跑 tests）
#       ./run.sh tests              只跑測試
#       ./run.sh gpt diffusion      只執行指定範例
# 範例：mlp cnn gru gpt diffusion（diffusion 在 CPU 上約需 7 分鐘）
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
examples=("$@")
[ ${#examples[@]} -eq 0 ] && examples=(tests mlp cnn gru gpt diffusion)

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
