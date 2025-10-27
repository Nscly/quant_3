#!/bin/bash

echo "=========================================="
echo "FinGAT 快速训练脚本"
echo "=========================================="
echo ""

# 检查Python环境
echo "[1] 检查Python环境..."
if command -v python3 &> /dev/null; then
    echo "✓ Python3 已安装: $(python3 --version)"
else
    echo "✗ Python3 未找到"
    exit 1
fi

# 检查数据文件
echo ""
echo "[2] 检查数据文件..."
if [ -f "Taiwan_model_data_10_best.pickle" ]; then
    echo "✓ 数据文件已存在"
else
    echo "✗ 数据文件不存在，正在运行数据准备..."
    python3 prepare_data.py
    if [ $? -ne 0 ]; then
        echo "✗ 数据准备失败"
        exit 1
    fi
fi

# 检查边矩阵文件
echo ""
echo "[3] 检查图边矩阵文件..."
EDGE_FILES=("Taiwan_inner_edge.npy" "Taiwan_outer_edge.npy" "edge_10.npy" "Taiwan_inner_edge20.npy")
ALL_EXIST=true

for file in "${EDGE_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file 存在"
    else
        echo "✗ $file 不存在"
        ALL_EXIST=false
    fi
done

if [ "$ALL_EXIST" = false ]; then
    echo ""
    echo "边矩阵文件缺失，运行数据准备脚本..."
    python3 prepare_data.py
fi

echo ""
echo "[4] 开始训练..."
echo "=========================================="
echo ""

# 默认使用CAT模型，CPU训练，10个epoch
MODEL=${1:-CAT}
DEVICE=${2:-cpu}
EPOCHS=${3:-10}

echo "配置:"
echo "  模型: $MODEL"
echo "  设备: $DEVICE"
echo "  轮数: $EPOCHS"
echo ""

python3 train.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --dim 16 \
    --lr 0.05 \
    --alpha 1 \
    --beta 1 \
    --gamma 1 \
    --week_num 3

echo ""
echo "=========================================="
echo "训练完成!"
echo "=========================================="
