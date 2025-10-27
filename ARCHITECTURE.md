# FinGAT 架构详解

## 🏗️ 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        FinGAT 系统架构                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  原始股票数据    │  27支股票 × 6个类别
│  (CSV文件)      │  - Bitcoin Mining
│                 │  - Semiconductors  
│                 │  - Energy
└────────┬────────┘  - Financials
         │           - Cloud/Big Tech
         │           - Data Infrastructure
         ↓
┌─────────────────────────────────────────────────────┐
│             数据预处理 (clean_data.py)               │
│  • 价格归一化                                         │
│  • 技术指标: 收益率、相对价格、移动平均                │
│  • 类别编码: One-hot                                  │
│  • 滑动窗口: 7天序列 × 4周                            │
└─────────────────────┬───────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│           图结构定义 (边矩阵)                         │
│                                                      │
│  Inner Edge:  同类别股票全连接                        │
│  ┌─────┐    ┌─────┐    ┌─────┐                      │
│  │Stock│────│Stock│────│Stock│  (类别内)             │
│  └─────┘    └─────┘    └─────┘                      │
│                                                      │
│  Outer Edge:  类别间连接                             │
│  ┌────────┐       ┌────────┐                        │
│  │Category│───────│Category│  (类别间)               │
│  └────────┘       └────────┘                        │
└─────────────────────┬───────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│              模型训练 (train.py)                      │
│                                                      │
│  输入: [Week1, Week2, Week3, Week4]                  │
│  每周: (N_stocks, 7_days, 30+_features)              │
│                                                      │
│  ┌─────────────────────────────────┐                │
│  │  模型选择 (--model参数)          │                │
│  │                                 │                │
│  │  • CG:     基础图注意力          │                │
│  │  • CAT:    双层图注意力 ⭐       │                │
│  │  • CPool:  图池化模型            │                │
│  └─────────────────────────────────┘                │
│                                                      │
│  多任务学习:                                          │
│  Loss = α×MAE + β×BCE + γ×Ranking                   │
│                                                      │
│  ↓                    ↓                  ↓           │
│  回归预测             分类预测            排序优化     │
│  (收益率)            (涨/跌)            (相对排序)    │
└─────────────────────┬───────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│              评估输出                                 │
│  • MAE:         回归误差                              │
│  • Accuracy:    涨跌准确率                            │
│  • MRR:         平均倒数排名                          │
│  • Precision@K: Top-K推荐精度                         │
└─────────────────────────────────────────────────────┘
```

## 📊 数据流详解

### 输入数据维度
```
训练数据:
  x1: (num_weeks, num_stocks, 7, num_features)  # Week 1
  x2: (num_weeks, num_stocks, 7, num_features)  # Week 2
  x3: (num_weeks, num_stocks, 7, num_features)  # Week 3
  x4: (num_weeks, num_stocks, 7, num_features)  # Week 4

其中:
  - num_weeks:    训练周数 (取决于数据量)
  - num_stocks:   股票数量 (27)
  - 7:            每周7天交易数据
  - num_features: 特征维度 (30+)

标签:
  y_return_ratio: (num_weeks, num_stocks)  # 第5周的收益率
  y_up_or_down:   (num_weeks, num_stocks)  # 第5周的涨跌 (0/1)
```

### 特征详解 (30+ 维度)
```
1. return_ratio     - 收益率
2. c_open           - 开盘价相对收盘价
3. c_high           - 最高价相对收盘价
4. c_low            - 最低价相对收盘价
5-10. [5,10,15,20,25,30]-days  - 移动平均线
11+. label_*        - 类别one-hot编码 (6维)
```

## 🧠 模型架构对比

### 1. CategoricalGraph (CG) - 基础模型
```
输入: 4周数据
  ↓
[SequenceEncoder × 4]  # 每周独立编码
  ↓ (GRU + Attention)
  ↓
Weekly Attention       # 融合4周信息
  ↓
Category GAT          # 类别级别图注意力
  ↓
Fusion Layer          # 特征融合
  ↓
Output: [Regression, Classification]
```

**特点:**
- 结构简单，训练快速
- 只考虑类别间关系
- 适合快速原型验证

### 2. CategoricalGraphAtt (CAT) - 推荐模型 ⭐
```
输入: 3-4周数据
  ↓
[SequenceEncoder × 3/4]
  ↓ GRU (可选)
  ↓
Weekly Attention
  ↓
┌─────────────┬──────────────┬─────────────┐
│  原始特征    │  Inner GAT   │  Outer GAT  │
│  (Weekly)   │  (股票关系)   │  (类别关系)  │
└─────────────┴──────────────┴─────────────┘
  ↓           ↓              ↓
  └───────────┴──────────────┘
              ↓
        3-way Fusion         # 三路特征融合
              ↓
  Output: [Regression, Classification]
```

**特点:**
- 双层图结构，捕获多尺度关系
- Inner GAT: 同类别股票相互影响
- Outer GAT: 不同类别间的联动
- 性能最佳，推荐使用

**关键参数:**
```python
--use_gru True   # 启用周级别GRU
--week_num 3     # 使用3周数据
--dim 16-32      # 隐藏层维度
```

### 3. CategoricalGraphPool (CPool) - 高级模型
```
输入: 3-4周数据
  ↓
[SequenceEncoder × 3/4]
  ↓
Weekly Attention
  ↓
Inner GAT              # 类别内交互
  ↓
SAGPooling × 5         # 每个类别独立池化
  ↓ (ratio=0.5)
[MaxPool, MeanPool]    # 全局池化
  ↓
Category GAT           # 类别级聚合
  ↓
3-way Fusion
  ↓
Output: [Regression, Classification]
```

**特点:**
- 包含图池化机制
- 自动学习重要股票
- 计算复杂度较高
- 适合研究和探索

## 🎯 损失函数设计

### 多任务联合损失
```python
Total_Loss = α × L_regression + β × L_classification + γ × L_ranking

其中:
  L_regression:    MAE (L1 Loss)
  L_classification: BCE (Binary Cross Entropy)
  L_ranking:       Pairwise Ranking Loss
```

### 1. 回归损失 (MAE)
```python
L_regression = |predicted_return - actual_return|
```
目标: 准确预测收益率

### 2. 分类损失 (BCE)
```python
L_classification = -[y·log(p) + (1-y)·log(1-p)]
```
目标: 准确预测涨跌方向

### 3. 排序损失 (Ranking)
```python
L_ranking = max(0, -(pred_i × pred_j) × (true_i × true_j))
```
目标: 保持相对排序一致
- 如果 true_i > true_j，则要求 pred_i > pred_j

### 损失权重调优建议
```
场景1: 重视预测精度
  α = 2.0, β = 1.0, γ = 0.5

场景2: 重视涨跌方向
  α = 0.5, β = 2.0, γ = 1.0

场景3: 重视排名推荐
  α = 1.0, β = 1.0, γ = 2.0

默认均衡:
  α = β = γ = 1.0
```

## 📈 训练流程

### 1. 初始化阶段
```python
# 1.1 加载数据
data = pickle.load("Taiwan_model_data_10_best.pickle")

# 1.2 加载图边矩阵
inner_edge = np.load("Taiwan_inner_edge.npy")
outer_edge = np.load("Taiwan_outer_edge.npy")

# 1.3 选择模型
if model == "CAT":
    model = CategoricalGraphAtt(...)

# 1.4 初始化参数 (Xavier)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
```

### 2. 训练循环
```python
for epoch in range(epochs):
    for week in range(num_weeks):
        # 2.1 准备批次数据
        batch = [x1[week], x2[week], x3[week], x4[week]]
        
        # 2.2 前向传播
        reg_out, cls_out = model(batch)
        
        # 2.3 计算损失
        loss = α×mae + β×bce + γ×ranking
        
        # 2.4 反向传播
        loss.backward()
        optimizer.step()
    
    # 2.5 验证评估
    evaluate_model()
```

### 3. 评估阶段
```python
# 3.1 预测
predictions = model.predict_toprank(test_data)

# 3.2 计算指标
mae = mean_absolute_error(y_true, y_pred)
acc = accuracy(y_true_class, y_pred_class)
mrr = mean_reciprocal_rank(y_true, y_pred, k=[5,10,20])
precision = precision_at_k(y_true, y_pred, k=[5,10,20])

# 3.3 保存最佳模型 (基于MRR)
if current_mrr > best_mrr:
    best_model = copy.deepcopy(model)
```

## 🔄 数据流转换

### 从CSV到模型输入
```
CSV (股票价格)
  ↓ [clean_data.py]
特征矩阵 (N_days × N_features)
  ↓ [滑动窗口]
周数据 (N_weeks × N_stocks × 7 × N_features)
  ↓ [SequenceEncoder]
序列嵌入 (N_weeks × N_stocks × hidden_dim)
  ↓ [Weekly Attention]
股票嵌入 (N_stocks × hidden_dim)
  ↓ [GAT]
增强嵌入 (N_stocks × hidden_dim)
  ↓ [Output Layer]
预测 (N_stocks × 1)
```

## 💡 关键技术点

### 1. 注意力机制
```python
# 时间步注意力
attention_weight = Linear(time_step → time_step)
attention_probs = softmax(attention_weight)
output = Σ(attention_probs × input)
```
作用: 自动学习重要的时间点

### 2. 图注意力网络 (GAT)
```python
# 节点更新
h_i' = Σ_j α_ij W h_j

其中:
  α_ij = attention(h_i, h_j)  # 边的注意力权重
  W    = 可学习的变换矩阵
```
作用: 聚合邻居节点信息

### 3. 双层图结构
```
Inner Graph (同类别):
  Mining股票A ←→ Mining股票B ←→ Mining股票C
  
Outer Graph (跨类别):
  Mining类别 ←→ Semiconductor类别 ←→ Energy类别
```

### 4. 多尺度特征融合
```python
fusion = concat([
    weekly_features,      # 原始时序特征
    inner_graph_features, # 类内关系特征
    outer_graph_features  # 类间关系特征
])
output = ReLU(Linear(fusion))
```

## 🎮 超参数影响分析

| 参数 | 增大影响 | 减小影响 | 推荐范围 |
|------|---------|---------|----------|
| `dim` | 模型容量↑, 计算慢↓ | 速度快↑, 性能可能下降↓ | 16-64 |
| `lr` | 收敛快↑, 可能震荡↓ | 稳定↑, 收敛慢↓ | 0.001-0.1 |
| `alpha` | 重视回归精度 | 忽视回归任务 | 0.5-2.0 |
| `beta` | 重视分类准确率 | 忽视分类任务 | 0.5-2.0 |
| `gamma` | 重视排序质量 | 忽视排序 | 0.1-2.0 |
| `week_num` | 历史信息多↑, 计算慢↓ | 信息少↓, 速度快↑ | 3-4 |
| `l2` | 正则化强↑, 可能欠拟合↓ | 可能过拟合↓ | 1e-6 to 1e-4 |

## 🔍 模型输出解读

### 预测输出
```python
reg_output: [stock1_return, stock2_return, ..., stock27_return]
  # 预测的收益率 (连续值)
  
cls_output: [stock1_prob, stock2_prob, ..., stock27_prob]
  # 上涨概率 (0-1之间)
```

### 推荐逻辑
```python
# 按预测收益率排序
ranked_stocks = sort(stocks, by=reg_output, descending=True)

# Top-K推荐
top_k_stocks = ranked_stocks[:k]

# 评估推荐质量
precision = len(intersection(
    top_k_predicted,
    top_k_actual
)) / k
```

## 🚀 性能优化技巧

### 1. 数据加载优化
- ✅ 预加载所有数据到GPU
- ✅ 使用.float()减少精度
- 🔲 可考虑数据并行

### 2. 模型优化
- ✅ Xavier初始化
- ✅ Dropout (0.2) 防过拟合
- 🔲 可添加BatchNorm
- 🔲 可添加残差连接

### 3. 训练优化
- ✅ Adam优化器
- 🔲 学习率调度 (StepLR, CosineAnnealing)
- 🔲 梯度裁剪 (防梯度爆炸)
- 🔲 早停机制

### 4. 内存优化
- 减小batch_size
- 减小hidden_dim
- 使用gradient checkpointing

## 📊 实验建议

### 快速验证 (5分钟)
```bash
python train.py --model CG --epochs 5 --device cpu
```

### 标准训练 (30分钟)
```bash
python train.py --model CAT --epochs 20 --dim 32 --device cuda:0
```

### 完整实验 (2小时)
```bash
python train.py --model CAT --epochs 100 --dim 64 \
  --lr 0.01 --l2 1e-5 --use_gru True --device cuda:0
```

## 🎓 理论基础

### 论文来源
- **FinGAT**: Financial Graph Attention Network to Recommend Top-K Profitable Stocks
- arXiv: 2106.10159

### 核心创新
1. 双层图结构建模股票关系
2. 多任务学习融合多个目标
3. 时序+图神经网络的组合

### 应用场景
- 股票推荐系统
- 量化交易策略
- 风险评估
- 投资组合优化

---

## 📝 总结

这个架构的核心优势:
1. **层次化建模**: Inner/Outer双层图捕获不同尺度的关系
2. **多任务学习**: 回归+分类+排序三个目标互相增强
3. **时序+图结合**: GRU处理时间序列 + GAT处理关系结构
4. **灵活可扩展**: 支持多种模型变体和参数配置

适用场景:
- 金融市场预测
- 关系型数据建模
- 推荐系统
- 时序预测任务
