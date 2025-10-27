# FinGAT 训练指南 - 修改后分支分析

## 📋 概述

这个修改后的branch是一个基于图注意力网络(Graph Attention Network)的金融股票推荐系统，专门用于预测和推荐加密矿工相关股票。该系统使用PyTorch和PyTorch Geometric实现，结合了时序建模(GRU)、注意力机制和图神经网络。

## 🏗️ 代码架构

### 1. 数据处理模块 (`clean_data.py`)
负责原始股票数据的预处理和特征工程

**主要功能：**
- 读取股票CSV文件和类别标签
- 价格归一化处理（StandardScaler）
- 技术指标计算：
  - 收益率（return ratio）
  - 相对价格特征（c_open, c_high, c_low）
  - 移动平均线（5/10/15/20/25/30天）
- 类别编码（One-hot encoding）
- 生成滑动窗口序列数据
- 输出pickle格式的训练/测试数据

### 2. 模型定义模块 (`model/graph_pool.py`)
定义了三种不同的图神经网络架构

**核心组件：**
- `AttentionBlock`: 时间步注意力机制
- `SequenceEncoder`: GRU + Attention的序列编码器
- `GraphEncoder`: GRU + GAT的图编码器

**三种模型架构：**

#### a) CategoricalGraph (CG)
- 最简单的基线模型
- 使用序列编码器处理每周数据
- 通过GAT学习类别间关系
- 融合时序和类别特征进行预测

#### b) CategoricalGraphAtt (CAT) ⭐ **推荐使用**
- 完整的双层图注意力网络
- **Inner GAT**: 学习同类别内股票之间的关系
- **Outer GAT**: 学习不同类别之间的关系
- 支持可选的GRU进行周级别时序建模
- 三路特征融合（weekly + category + inner）

#### c) CategoricalGraphPool (CPool)
- 包含图池化机制的高级模型
- 使用SAGPooling进行图节点采样
- 结合全局最大池化和平均池化
- 计算复杂度较高

### 3. 训练模块 (`train.py`)
主训练脚本，实现多任务学习

**损失函数：**
```python
Loss = α × MAE_Loss + β × BCE_Loss + γ × Ranking_Loss
```

- **MAE Loss**: 回归损失，预测收益率
- **BCE Loss**: 二分类损失，预测涨跌
- **Ranking Loss**: 排序损失，确保相对排序正确

**评估指标：**
- MAE (Mean Absolute Error): 回归精度
- Accuracy: 涨跌预测准确率
- MRR (Mean Reciprocal Rank): 平均倒数排名
- Precision@K: Top-K推荐精度

### 4. 参数配置模块 (`parse_arg.py`)
命令行参数解析

## 🚀 训练流程详解

### 步骤1: 准备数据

**当前数据情况分析：**
```
数据源：stock_data/ 目录
- 27支股票的历史价格数据（CSV格式）
- 涵盖6个类别：
  1. Bitcoin Mining (12支)
  2. Semiconductors (4支)
  3. Energy (3支)
  4. Financials - Traditional (3支)
  5. Cloud / Big Tech (3支)
  6. Data Infrastructure (2支)

类别标签：crypto_miners_category.csv
```

**需要修改的配置：**

`clean_data.py` 需要更新以适配当前数据：

```python
# 第9行：更新类别文件名
SP500_name = pd.read_csv("crypto_miners_category.csv")

# 第17行：更新数据目录
da['stock_price'] = pd.read_csv("./stock_data/%s.csv"%(target))
```

**运行数据预处理：**
```bash
python clean_data.py
```

此步骤会生成必要的数据文件：
- `Taiwan_model_data_10_best.pickle`: 训练和测试数据
- 边矩阵文件（需要手动创建或从其他源获取）：
  - `Taiwan_inner_edge.npy`: 内部边（同类别股票连接）
  - `edge_10.npy`: 10邻居边
  - `Taiwan_inner_edge20.npy`: 20邻居边
  - `Taiwan_outer_edge.npy`: 外部边（类别间连接）

### 步骤2: 训练模型

**基础训练命令：**

```bash
# 使用CAT模型（推荐）
python train.py --model CAT --epochs 20 --dim 16 --lr 0.05

# 使用CPU训练
python train.py --model CAT --device cpu --epochs 10

# 使用GPU训练
python train.py --model CAT --device cuda:0 --epochs 20
```

**完整参数调优示例：**

```bash
python train.py \
  --model CAT \
  --epochs 50 \
  --dim 32 \
  --lr 0.01 \
  --l2 1e-6 \
  --alpha 1.0 \
  --beta 1.0 \
  --gamma 0.5 \
  --week_num 3 \
  --use_gru True \
  --device cuda:0
```

**关键超参数说明：**

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `--model` | CAT | 模型选择 (CG/CAT/CPool) | CAT性能最好 |
| `--epochs` | 20 | 训练轮数 | 20-50 |
| `--dim` | 16 | 隐藏层维度 | 16-64，越大越慢 |
| `--lr` | 0.05 | 学习率 | 0.001-0.1 |
| `--l2` | 0 | L2正则化 | 1e-6 到 1e-4 |
| `--alpha` | 1 | MAE损失权重 | 0.5-2.0 |
| `--beta` | 1 | 分类损失权重 | 0.5-2.0 |
| `--gamma` | 1 | 排序损失权重 | 0.1-1.0 |
| `--week_num` | 3 | 输入周数 | 3-4 |
| `--use_gru` | False | 是否使用GRU | 建议开启 |
| `--device` | cuda:1 | 设备 | cpu或cuda:0 |

### 步骤3: 训练过程

**训练循环：**
```python
for epoch in range(epochs):
    for week in range(num_weeks):
        # 1. 前向传播
        reg_out, cls_out = model(batch_weekly)
        
        # 2. 计算损失
        reg_loss = MAE(reg_out, batch_reg_y)
        cls_loss = BCE(cls_out, batch_cls_y)
        rank_loss = RankingLoss(reg_out, batch_reg_y)
        
        # 3. 反向传播和优化
        total_loss = α×reg_loss + β×cls_loss + γ×rank_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    # 4. 验证评估
    evaluate_on_test_set()
```

**输出示例：**
```
Evaluate at epoch 1
REG Loss:0.0234 CLS Loss:0.4521 RANK Loss:12.3456  Loss:12.8211
[[MAE, ACC, MRR, Precision@5],
 [MAE, ACC, MRR, Precision@10],
 [MAE, ACC, MRR, Precision@20]]

-------Final result-------
[BEST MRR] MAE:0.0234 ACC:0.6234 MRR:0.7543 Precision:0.6800
```

## 📊 数据流程图

```
原始数据 (CSV) 
    ↓
clean_data.py (特征工程)
    ↓
Pickle文件 (训练/测试张量)
    ↓
train.py 加载数据
    ↓
模型训练 (多任务损失)
    ↓
评估指标 (MAE, ACC, MRR, Precision)
    ↓
最佳模型
```

## 🔧 常见问题和解决方案

### 问题1: 缺失边矩阵文件
**错误信息：**
```
FileNotFoundError: [Errno 2] No such file or directory: './Taiwan_inner_edge.npy'
```

**解决方案：**
需要手动创建图的边矩阵。边矩阵定义了节点（股票）之间的连接关系。

```python
import numpy as np

# 示例：创建全连接的inner edge（同类别内）
# 假设每个类别有20支股票，共5个类别（100支股票）
edges = []
for category in range(5):
    stocks = range(category*20, (category+1)*20)
    for i in stocks:
        for j in stocks:
            if i != j:
                edges.append([i, j])
                
inner_edge = np.array(edges)
np.save("Taiwan_inner_edge.npy", inner_edge)

# Outer edge连接不同类别的代表节点
outer_edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 4]]  # 示例
np.save("Taiwan_outer_edge.npy", np.array(outer_edges))
```

### 问题2: 数据形状不匹配
确保clean_data.py正确生成数据后再训练。

### 问题3: CUDA内存不足
- 减小 `--dim` 参数
- 使用 `--device cpu`
- 减少batch累积的week数量

### 问题4: 收敛速度慢
- 调整学习率 `--lr`
- 调整损失函数权重 `--alpha`, `--beta`, `--gamma`
- 增加 `--dim` 提高模型容量

## 💡 最佳实践建议

1. **首次训练：** 使用默认参数快速验证流程
   ```bash
   python train.py --model CAT --device cpu --epochs 5
   ```

2. **超参数搜索：** 系统地调整关键参数
   - 学习率：[0.001, 0.01, 0.05, 0.1]
   - 隐藏维度：[16, 32, 64]
   - 损失权重：网格搜索

3. **模型选择：**
   - 快速实验：CG (最快)
   - 生产部署：CAT (最准)
   - 研究尝试：CPool (最新)

4. **评估重点：**
   - 回归任务：关注MAE
   - 推荐任务：关注MRR和Precision
   - 交易策略：关注涨跌准确率

## 📈 性能优化

1. **数据加载优化：**
   - 预加载所有数据到GPU
   - 当前已实现批量处理

2. **模型优化：**
   - Xavier初始化已实现
   - Dropout防止过拟合 (0.2)

3. **训练优化：**
   - Adam优化器
   - 可考虑学习率调度

## 🎯 输出和结果

训练完成后会输出：
- 每个epoch的损失值
- 测试集上的多个指标
- 最佳MRR对应的完整结果

可以根据需要添加：
- 模型保存（checkpoint）
- TensorBoard可视化
- 预测结果导出

## 📝 代码修改建议

为了使代码能够运行，建议以下修改：

1. **更新clean_data.py的文件路径**
2. **创建或生成边矩阵文件**
3. **添加模型保存功能**
4. **添加更详细的日志**

## 🔍 下一步开发

- [ ] 实现模型checkpoint保存
- [ ] 添加早停机制
- [ ] 实现数据增强
- [ ] 添加更多评估指标
- [ ] 可视化attention权重
- [ ] 实时预测接口

---

**总结：** 这是一个设计良好的多任务学习系统，结合了深度学习、图神经网络和金融特征工程。主要挑战是数据准备和图结构定义。建议从简单的CAT模型开始，逐步优化参数和架构。
