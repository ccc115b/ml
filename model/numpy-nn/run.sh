#!/usr/bin/env bash
# 依序執行測試與所有範例；任何一個失敗就停止。
# 用法：./run.sh              執行全部（先跑 tests）
#       ./run.sh tests        只跑測試
#       ./run.sh gpt gru      只執行指定範例
# 範例：mlp cnn gru gpt
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
examples=("$@")
[ ${#examples[@]} -eq 0 ] && examples=(tests mlp cnn gru gpt)

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
