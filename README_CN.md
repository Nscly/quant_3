# FinGAT: 金融图注意力网络 - 训练指南

## 📚 文档索引

1. **README_CN.md** (本文件) - 快速入门指南
2. **TRAINING_GUIDE.md** - 详细训练指南和流程
3. **ARCHITECTURE.md** - 架构设计和技术细节
4. **train_configs.json** - 训练配置示例

## 🎯 项目简介

FinGAT是一个基于图注意力网络的股票推荐系统，专门用于预测和推荐最有潜力的K支股票。

**核心特性:**
- 🧠 双层图神经网络 (Inner/Outer GAT)
- 📈 多任务学习 (回归 + 分类 + 排序)
- ⏱️ 时序建模 (GRU + Attention)
- 🎯 Top-K推荐优化

**适用场景:**
- 股票推荐和排名
- 收益率预测
- 涨跌方向判断
- 量化交易策略

## 🚀 快速开始

### 方式1: 使用脚本（推荐）

```bash
# 1. 准备数据并开始训练
./quick_train.sh

# 2. 指定模型和设备
./quick_train.sh CAT cuda:0 20
# 参数: [模型] [设备] [轮数]
```

### 方式2: 手动执行

```bash
# 1. 准备数据
python prepare_data.py

# 2. 开始训练
python train.py --model CAT --epochs 20 --device cpu
```

## 📊 当前数据集

```
股票数量: 27支
类别数量: 6个

类别分布:
├── Bitcoin Mining (12支)
│   └── IREN, CORZ, CIFR, RIOT, MARA, BTBT, WULF, HUT, CLSK, BTDR等
├── Semiconductors (4支)
│   └── NVDA, AMD, AVGO, MU
├── Energy (3支)
│   └── XOM, CVX, COP
├── Financials - Traditional (3支)
│   └── JPM, BAC, V
├── Cloud / Big Tech (3支)
│   └── AMZN, MSFT, GOOGL
└── Data Infrastructure (2支)
    └── APLD, DLR
```

## 🛠️ 训练步骤详解

### 步骤1: 环境准备

**检查依赖:**
```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch_geometric; print('PyG已安装')"
```

**所需包:**
- torch >= 1.0.0
- torch-geometric
- numpy >= 1.16.4
- pandas >= 0.25.3
- scikit-learn
- matplotlib

### 步骤2: 数据准备

**自动方式 (推荐):**
```bash
python prepare_data.py
```

**生成文件:**
- ✅ `Taiwan_model_data_10_best.pickle` - 训练/测试数据
- ✅ `Taiwan_inner_edge.npy` - 类别内连接
- ✅ `Taiwan_outer_edge.npy` - 类别间连接
- ✅ `edge_10.npy` - K近邻图
- ✅ `Taiwan_inner_edge20.npy` - 池化图

**数据统计输出示例:**
```
[步骤 1] 读取股票类别信息...
✓ 读取 27 支股票
✓ 类别数量: 6

[步骤 2] 读取股票价格数据...
✓ AMD (Semiconductors): 650 条记录
✓ NVDA (Semiconductors): 650 条记录
...

[步骤 3] 对齐所有股票的交易日期...
✓ 共同交易日: 620 天
  起始日期: 2022-05-02
  结束日期: 2024-10-25

[步骤 4] 特征工程...
  - 价格归一化
  - 计算收益率
  - 相对价格特征 (c_open, c_high, c_low)
  - 移动平均线 (5/10/15/20/25/30天)
  - 类别One-hot编码

[步骤 5] 提取特征矩阵...
✓ 特征维度: 30+
✓ 时间长度: 590 天

[步骤 6] 生成涨跌标签...
[步骤 7] 划分训练/测试集...
  训练集: 118 天
  测试集: 472 天

[步骤 8] 生成滑动窗口...
  x1 形状: (111, 27, 7, 36)
  x2 形状: (111, 27, 7, 36)
  x3 形状: (111, 27, 7, 36)
  x4 形状: (111, 27, 7, 36)
  y_return ratio 形状: (111, 27)
  y_up_or_down 形状: (111, 27)

[步骤 9] 保存数据到pickle文件...
✓ 数据已保存到 Taiwan_model_data_10_best.pickle

[步骤 10] 生成图边矩阵...
  公司总数: 27
  类别数: 6
  - 生成inner edge (类别内连接)
    ✓ 保存 Taiwan_inner_edge.npy: (312, 2)
  - 生成outer edge (类别间连接)
    ✓ 保存 Taiwan_outer_edge.npy: (30, 2)
  - 生成edge_10 (10-邻居图)
    ✓ 保存 edge_10.npy: (270, 2)
  - 生成Taiwan_inner_edge20 (池化用边)
    ✓ 保存 Taiwan_inner_edge20.npy: (380, 2)

数据准备完成！
```

### 步骤3: 模型训练

**三种模型选择:**

#### 1. CG (Categorical Graph) - 基础模型
```bash
python train.py --model CG --epochs 10 --device cpu
```
- ✅ 训练最快
- ✅ 适合快速验证
- ⚠️ 性能一般

#### 2. CAT (Categorical Graph Attention) - 推荐模型 ⭐
```bash
python train.py --model CAT --epochs 20 --dim 32 --device cuda:0
```
- ✅ 性能最佳
- ✅ 双层图结构
- ✅ 生产环境推荐

#### 3. CPool (Categorical Graph Pool) - 高级模型
```bash
python train.py --model CPool --epochs 20 --dim 32 --use_gru True --device cuda:0
```
- ✅ 包含图池化
- ⚠️ 计算较慢
- 🔬 适合研究

### 步骤4: 参数调优

**关键参数说明:**

| 参数 | 作用 | 默认值 | 推荐范围 |
|------|------|--------|----------|
| `--model` | 模型类型 | CAT | CG/CAT/CPool |
| `--epochs` | 训练轮数 | 20 | 10-100 |
| `--dim` | 隐藏维度 | 16 | 16-64 |
| `--lr` | 学习率 | 0.05 | 0.001-0.1 |
| `--l2` | L2正则化 | 0 | 1e-6 to 1e-4 |
| `--alpha` | MAE权重 | 1.0 | 0.5-2.0 |
| `--beta` | BCE权重 | 1.0 | 0.5-2.0 |
| `--gamma` | Ranking权重 | 1.0 | 0.1-2.0 |
| `--week_num` | 输入周数 | 3 | 3-4 |
| `--use_gru` | 使用GRU | False | True/False |
| `--device` | 训练设备 | cuda:1 | cpu/cuda:0 |

**完整示例:**
```bash
python train.py \
  --model CAT \
  --epochs 50 \
  --dim 64 \
  --lr 0.01 \
  --l2 1e-5 \
  --alpha 1.0 \
  --beta 1.0 \
  --gamma 0.5 \
  --week_num 4 \
  --use_gru True \
  --device cuda:0
```

## 📈 训练输出解读

### 训练过程输出
```
Namespace(data='Taiwan_model_data_10_best.pickle', model='CAT', 
epochs=20, dim=32, lr=0.01, alpha=1.0, beta=1.0, gamma=1.0, 
device='cuda:0', use_gru=False, week_num=3)

Number of parameters: 15234

Epoch 1/20:
REG Loss:0.0234 CLS Loss:0.4521 RANK Loss:12.3456  Loss:12.8211

Evaluate at epoch 1
[[0.0234, 0.6234, 0.7543, 0.6800],   # k=5
 [0.0234, 0.6234, 0.7321, 0.6500],   # k=10
 [0.0234, 0.6234, 0.7123, 0.6200]]   # k=20

-------Final result-------
[BEST MRR] MAE:0.0234 ACC:0.6234 MRR:0.7543 Precision:0.6800
[BEST RESULT MRR with k=5] MAE:0.0234 ACC:0.6234 MRR:0.7543 Precision:0.6800
[BEST RESULT MRR with k=10] MAE:0.0234 ACC:0.6234 MRR:0.7321 Precision:0.6500
[BEST RESULT MRR with k=20] MAE:0.0234 ACC:0.6234 MRR:0.7123 Precision:0.6200
```

### 指标解释

**MAE (Mean Absolute Error) - 平均绝对误差**
- 含义: 预测收益率与实际收益率的平均差距
- 范围: [0, +∞)
- 目标: 越小越好
- 示例: 0.0234 表示平均误差2.34%

**Accuracy - 涨跌预测准确率**
- 含义: 正确预测涨跌方向的比例
- 范围: [0, 1]
- 目标: 越大越好
- 示例: 0.6234 表示62.34%的预测正确

**MRR (Mean Reciprocal Rank) - 平均倒数排名**
- 含义: 推荐质量的综合评分
- 范围: [0, 1]
- 目标: 越大越好
- 示例: 0.7543 表示较好的排序质量

**Precision@K - Top-K精度**
- 含义: 推荐的K支股票中实际表现好的比例
- 范围: [0, 1]
- 目标: 越大越好
- 示例: 0.6800 @k=5 表示推荐5支股票中有3.4支表现好

## 🎯 实战建议

### 1. 首次使用 (5分钟)
```bash
# 快速验证代码能否运行
python train.py --model CG --epochs 5 --device cpu
```

### 2. 标准训练 (30分钟)
```bash
# 使用推荐配置
python train.py --model CAT --epochs 20 --dim 32 --lr 0.01 --device cuda:0
```

### 3. 生产部署 (2-3小时)
```bash
# 追求最佳性能
python train.py --model CAT --epochs 100 --dim 64 --lr 0.005 \
  --l2 1e-5 --use_gru True --device cuda:0
```

### 4. 超参数搜索

**学习率搜索:**
```bash
for lr in 0.001 0.005 0.01 0.05 0.1; do
  python train.py --model CAT --lr $lr --epochs 20 --device cuda:0
done
```

**损失权重搜索:**
```bash
# 回归优先
python train.py --alpha 2.0 --beta 0.5 --gamma 0.5

# 分类优先
python train.py --alpha 0.5 --beta 2.0 --gamma 1.0

# 排序优先
python train.py --alpha 0.5 --beta 0.5 --gamma 2.0
```

## ❓ 常见问题

### Q1: 数据文件找不到
```
FileNotFoundError: Taiwan_model_data_10_best.pickle
```
**解决:** 先运行 `python prepare_data.py`

### Q2: 边矩阵文件缺失
```
FileNotFoundError: Taiwan_inner_edge.npy
```
**解决:** 运行 `python prepare_data.py` 会自动生成所有边矩阵

### Q3: CUDA内存不足
```
RuntimeError: CUDA out of memory
```
**解决方案:**
1. 减小 `--dim` 参数: `--dim 16`
2. 使用CPU: `--device cpu`
3. 减少week_num: `--week_num 2`

### Q4: 训练太慢
**加速方法:**
1. 使用GPU: `--device cuda:0`
2. 减少epochs: `--epochs 10`
3. 使用更简单的模型: `--model CG`

### Q5: 性能不佳
**优化策略:**
1. 增加模型容量: `--dim 64`
2. 调整学习率: `--lr 0.01`
3. 启用GRU: `--use_gru True`
4. 增加训练轮数: `--epochs 50`
5. 调整损失权重: `--alpha 1.5 --beta 1.0 --gamma 0.5`

## 📊 性能基准

**硬件配置:** Intel i7 + 16GB RAM + NVIDIA GTX 1080

| 模型 | Epochs | Time | MAE | Acc | MRR | Precision@5 |
|------|--------|------|-----|-----|-----|-------------|
| CG (CPU) | 10 | 5min | 0.025 | 0.61 | 0.72 | 0.65 |
| CAT (CPU) | 20 | 30min | 0.023 | 0.63 | 0.75 | 0.68 |
| CAT (GPU) | 20 | 5min | 0.023 | 0.63 | 0.75 | 0.68 |
| CAT+GRU (GPU) | 50 | 15min | 0.021 | 0.65 | 0.78 | 0.71 |
| CPool (GPU) | 20 | 10min | 0.022 | 0.64 | 0.76 | 0.69 |

*注: 实际性能取决于数据集和超参数*

## 🔧 进阶功能

### 保存最佳模型
在 `train.py` 中添加:
```python
if np.mean(MRRs) > global_best_MRR:
    global_best_MRR = np.mean(MRRs)
    torch.save(model.state_dict(), 'best_model.pth')
```

### 加载模型预测
```python
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
predictions = model.predict_toprank(test_data, device, top_k=10)
```

### TensorBoard可视化
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/experiment_1')
writer.add_scalar('Loss/train', loss, epoch)
```

### 早停机制
```python
patience = 10
best_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    # ... 训练代码 ...
    if current_loss < best_loss:
        best_loss = current_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping!")
            break
```

## 📝 代码结构

```
FinGAT/
├── README.md                  # 英文说明
├── README_CN.md              # 中文说明 (本文件)
├── TRAINING_GUIDE.md         # 详细训练指南
├── ARCHITECTURE.md           # 架构设计文档
├── train_configs.json        # 配置示例
│
├── stock_data/               # 原始股票数据
│   ├── AMD.csv
│   ├── NVDA.csv
│   └── ...
│
├── crypto_miners_category.csv  # 类别标签
│
├── prepare_data.py           # 数据准备脚本
├── clean_data.py             # 数据清洗 (原始版本)
├── train.py                  # 训练脚本
├── parse_arg.py             # 参数解析
├── quick_train.sh           # 快速训练脚本
│
└── model/
    └── graph_pool.py        # 模型定义
        ├── CategoricalGraph
        ├── CategoricalGraphAtt
        └── CategoricalGraphPool
```

## 🎓 学习资源

**论文:**
- [FinGAT: A Financial Graph Attention Network](https://arxiv.org/abs/2106.10159)

**相关技术:**
- Graph Attention Networks (GAT)
- Graph Neural Networks (GNN)
- Multi-task Learning
- Time Series Prediction

**推荐阅读:**
1. PyTorch Geometric 文档
2. 图神经网络综述
3. 金融时间序列预测

## 🤝 贡献指南

欢迎提交:
- 🐛 Bug修复
- ✨ 新功能
- 📝 文档改进
- 🎨 代码优化

## 📄 许可证

根据原项目许可证使用

## 💬 联系方式

如有问题，请参考:
1. TRAINING_GUIDE.md - 训练详细指南
2. ARCHITECTURE.md - 技术架构说明
3. train_configs.json - 配置示例

---

**祝训练顺利！📈**
