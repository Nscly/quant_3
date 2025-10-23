# 快速开始指南 (Quick Start Guide)

## 🚀 5分钟快速启动

### 步骤 1: 设置虚拟环境

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# Windows: venv\Scripts\activate
```

### 步骤 2: 安装依赖包

```bash
pip install -r requirements.txt
```

需要安装的包：
- pandas (>=2.0.0) - 数据处理
- numpy (>=1.16.4) - 数值计算
- scikit-learn (>=1.0.0) - 机器学习工具
- torch (>=1.0.0) - 深度学习框架
- torch-geometric (>=2.0.0) - 图神经网络
- matplotlib (>=3.0.0) - 可视化

### 步骤 3: 数据预处理

```bash
python clean_data.py
```

这将生成：
- ✅ `stock_data_processed.pickle` - 处理后的数据
- ✅ `inner_edge.npy` - 类内图边
- ✅ `inner10_edge.npy` - Top-10相关性边
- ✅ `inner20_edge.npy` - Top-20相关性边  
- ✅ `outer_edge.npy` - 类间图边
- ✅ `category_mapping.pickle` - 类别映射

### 步骤 4: 训练模型

```bash
# 基础训练（推荐初学者）
python train.py --model CAT --epochs 20 --device cpu

# 快速测试（1个epoch）
python train.py --model CAT --epochs 1 --device cpu

# 高级训练（自定义参数）
python train.py --model CAT --epochs 50 --lr 0.01 --dim 32 --alpha 2 --beta 1 --gamma 0.5
```

### 步骤 5: 验证安装

```bash
python test_models.py
```

---

## 📊 数据集信息

当前数据集包含：
- **27只股票** 分布在 **8个类别**
- **训练集**: 52个时间窗口
- **测试集**: 6个时间窗口
- **特征维度**: 19维
- **时间步长**: 7天（周）

类别分布：
```
Bitcoin Mining        : 10只股票
Semiconductors        : 4只股票
Energy               : 3只股票
Financials - Trad.   : 3只股票
Cloud / Big Tech     : 3只股票
Data Infrastructure  : 2只股票
Auto Finance         : 1只股票
Financials Crypto    : 1只股票
```

---

## 🎯 模型选择

### CG (Categorical Graph)
- 基础模型
- 参数量: ~6,374
- 速度最快
```bash
python train.py --model CG
```

### CAT (Categorical Graph with Attention) ⭐ 推荐
- 带注意力机制
- 参数量: ~6,678
- 性能最佳
```bash
python train.py --model CAT
```

### CPool (Categorical Graph with Pooling)
- 带池化层
- 参数量: ~6,712
- 适合大规模数据
```bash
python train.py --model CPool
```

---

## 🔧 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | CAT | 模型类型 (CG/CAT/CPool) |
| `--epochs` | 20 | 训练轮数 |
| `--lr` | 0.05 | 学习率 |
| `--dim` | 16 | 隐藏层维度 |
| `--alpha` | 1 | MAE损失权重 |
| `--beta` | 1 | 分类损失权重 |
| `--gamma` | 1 | 排序损失权重 |
| `--device` | cpu | 设备 (cpu/cuda:0) |
| `--week_num` | 3 | 聚合周数 |

---

## 📈 评估指标

训练过程会输出以下指标：

- **MAE** (Mean Absolute Error): 收益率预测误差
- **Accuracy**: 涨跌方向预测准确率
- **MRR** (Mean Reciprocal Rank): 平均倒数排名
- **Precision@K**: K值推荐精确度 (K=5,10,20)

---

## ⚡ GPU 加速（可选）

如果有NVIDIA GPU：

```bash
# 安装GPU版本的PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 使用GPU训练
python train.py --model CAT --epochs 20 --device cuda:0
```

---

## 🐛 常见问题

### 问题1: ModuleNotFoundError
**解决**: 确保激活了虚拟环境并安装了所有依赖
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 问题2: 找不到数据文件
**解决**: 先运行数据预处理
```bash
python clean_data.py
```

### 问题3: CUDA out of memory
**解决**: 使用CPU或减小batch size
```bash
python train.py --device cpu
```

### 问题4: 训练太慢
**解决**: 减少epochs或使用GPU
```bash
python train.py --epochs 5 --device cpu
```

---

## 📚 更多信息

- 详细变更: 查看 [CHANGES.md](CHANGES.md)
- 完整文档: 查看 [README.md](README.md)
- 问题报告: GitHub Issues

---

## ✅ 验证清单

在开始训练前，确保：

- [ ] 虚拟环境已创建并激活
- [ ] 所有依赖包已安装 (`pip list`)
- [ ] 数据预处理已完成 (生成了 .pickle 和 .npy 文件)
- [ ] 测试脚本通过 (`python test_models.py`)

全部完成后即可开始训练！🎉

---

**提示**: 首次运行建议使用 `--epochs 1` 快速验证流程，确认无误后再进行完整训练。
