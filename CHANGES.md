# 代码修复和改进总结

## 环境配置和依赖安装

### 推荐使用虚拟环境

为了避免包冲突，强烈建议在虚拟环境中运行此项目：

#### 方法1：使用 venv（Python内置）

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
# Linux/Mac:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 退出虚拟环境（完成工作后）
deactivate
```

#### 方法2：使用 conda

```bash
# 创建虚拟环境
conda create -n fingat python=3.8

# 激活虚拟环境
conda activate fingat

# 安装依赖
pip install -r requirements.txt

# 退出虚拟环境
conda deactivate
```

### 必需的Python包

以下是项目依赖的所有包（已包含在 `requirements.txt` 中）：

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| `pandas` | >=2.0.0 | 数据处理和分析 |
| `numpy` | >=1.16.4 | 数值计算和数组操作 |
| `scikit-learn` | >=1.0.0 | 数据标准化和评估指标 |
| `torch` | >=1.0.0 | 深度学习框架 |
| `torch-geometric` | >=2.0.0 | 图神经网络库 |
| `matplotlib` | >=3.0.0 | 可视化（可选） |

### 安装步骤

```bash
# 1. 克隆仓库
git clone <repository-url>
cd <repository-name>

# 2. 创建并激活虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 数据预处理
python clean_data.py

# 5. 训练模型
python train.py --model CAT --epochs 20 --device cpu
```

### CPU vs GPU 训练

- **CPU训练**（默认）：适合小规模测试和开发
  ```bash
  python train.py --device cpu
  ```

- **GPU训练**：需要CUDA支持，训练速度更快
  ```bash
  # 确保安装了GPU版本的PyTorch
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  
  # 使用GPU训练
  python train.py --device cuda:0
  ```

### 验证安装

运行测试脚本验证所有依赖都已正确安装：

```bash
python test_models.py
```

如果所有测试通过，说明环境配置成功！

---

## 主要问题及解决方案

### 1. ⭐ 图构建功能缺失（最关键）
**问题**: 代码中完全缺少图结构的构建逻辑
**解决方案**:
- 在 `clean_data.py` 中添加 `build_graph_edges()` 函数
- 实现了三种类型的图边:
  - **Inner edges (类内边)**: 同一类别内的股票全连接
  - **Inner10/Inner20 edges**: 基于收益率相关性的Top-K连接
  - **Outer edges (类间边)**: 不同类别之间的全连接
- 使用相关系数矩阵计算股票之间的关联性
- 自动保存图边矩阵为 `.npy` 文件

### 2. 文件路径和命名不匹配
**问题**: 
- `clean_data.py` 读取 "SP500_category.csv" 但实际文件是 `crypto_miners_category.csv`
- 读取 "./SP500_dataset/" 但实际目录是 `./stock_data/`
- `train.py` 期望加载 "Taiwan_model_data_10_best.pickle" 等台湾股市相关文件

**解决方案**:
- 修改文件路径指向实际存在的文件
- 更新默认数据文件名为 `stock_data_processed.pickle`
- 更新图边文件名去掉 "Taiwan" 前缀

### 3. 数据处理问题
**问题**:
- `clean_data.py` 没有保存处理后的数据
- 训练/测试数据集拆分比例错误（train_size=0.2, test_size=0.8）

**解决方案**:
- 添加数据保存逻辑，生成 pickle 文件
- 修正拆分比例为 80% 训练 / 20% 测试
- 添加详细的处理进度和统计信息输出

### 4. 模型架构硬编码问题
**问题**:
- 模型假设固定形状 (5, 20) - 5个类别，每个20只股票
- `CategoricalGraphAtt` 中硬编码 `view(5,20,-1)` 和 `expand(-1,20,-1)`
- 实际数据有8个类别，27只股票

**解决方案**:
- 简化模型架构，移除硬编码的形状假设
- 使用动态的股票数量和类别数量
- 保留核心的GRU+Attention+GAT架构
- 确保模型可以处理任意数量的股票和类别

### 5. 数据一致性问题
**问题**:
- 不同股票的数据日期可能不对齐
- 某些股票数据长度不同（如 BTDR 只有633行，其他有872行）

**解决方案**:
- 使用最短股票数据的日期作为基准
- 确保所有股票使用相同的日期范围
- 避免因数据长度不一致导致的错误

### 6. 缺少配置和文档
**问题**:
- 缺少 requirements.txt
- 缺少 .gitignore
- README 过于简单

**解决方案**:
- 创建完整的 `requirements.txt`
- 创建合适的 `.gitignore`
- 大幅更新 README.md 添加详细使用说明
- 创建测试脚本 `test_models.py`

## 新增功能

### 1. 图构建算法
```python
def build_graph_edges(category_df, stock_data):
    # 1. 类内全连接图
    # 2. 基于相关性的Top-K图
    # 3. 类间全连接图
    return inner_edge, inner10_edge, inner20_edge, outer_edge
```

### 2. 数据预处理管道
- 自动对齐不同股票的日期
- 特征工程（移动平均、收益率等）
- One-hot编码类别特征
- 滑动窗口创建训练样本

### 3. 灵活的模型架构
- 支持动态数量的股票和类别
- 三种模型变体：CG, CAT, CPool
- 可配置的超参数

### 4. 完整的评估指标
- MAE (Mean Absolute Error)
- Accuracy (分类准确率)
- MRR (Mean Reciprocal Rank)
- Precision@K (K=5, 10, 20)

## 生成的文件

### 数据文件
- `stock_data_processed.pickle` - 处理后的训练/测试数据
- `inner_edge.npy` - 类内图边 (122条边)
- `inner10_edge.npy` - Top-10相关性边 (270条边)
- `inner20_edge.npy` - Top-20相关性边 (540条边)
- `outer_edge.npy` - 类间图边 (56条边)
- `category_mapping.pickle` - 类别映射信息

### 配置文件
- `requirements.txt` - Python依赖
- `.gitignore` - Git忽略规则
- `CHANGES.md` - 本文档
- `test_models.py` - 模型测试脚本

## 数据统计

当前数据集包含:
- **27只股票** 分布在 **8个类别**中:
  - Bitcoin Mining: 10只
  - Semiconductors: 4只
  - Energy: 3只
  - Financials - Traditional: 3只
  - Cloud / Big Tech: 3只
  - Data Infrastructure: 2只
  - Auto Finance: 1只
  - Financials Crypto: 1只

- **训练数据**: 52个时间窗口
- **测试数据**: 6个时间窗口
- **特征维度**: 19 (包括价格特征、移动平均、类别编码等)
- **时间步长**: 7天 (每周)

## 使用方法

### 第一步：数据预处理
```bash
python clean_data.py
```

### 第二步：训练模型
```bash
# 使用默认参数
python train.py

# 自定义参数
python train.py --model CAT --epochs 20 --lr 0.01 --dim 32 --device cpu
```

### 测试所有模型
```bash
python test_models.py
```

## 验证结果

所有三个模型都已成功测试:
- ✓ CG (Categorical Graph)
- ✓ CAT (Categorical Graph with Attention) - **推荐**
- ✓ CPool (Categorical Graph with Pooling)

测试结果显示模型能够:
- 正确加载数据和图结构
- 完成前向传播和反向传播
- 计算所有损失函数
- 生成预测并计算评估指标

## 性能参数

- CG模型参数量: 6,374
- CAT模型参数量: 6,678 
- CPool模型参数量: 6,712

初步测试结果(1个epoch):
- MAE: ~0.03-0.04
- Accuracy: ~0.41
- MRR@20: ~2.53
- Precision@20: ~0.72

## 技术要点

### 图注意力网络 (GAT)
- 使用PyTorch Geometric的GATConv层
- 捕获股票之间的关系和影响
- 自适应学习注意力权重

### 多任务学习
- 回归任务: 预测收益率
- 分类任务: 预测涨跌方向
- 排序任务: 学习相对排序

### 损失函数
```python
loss = alpha * MAE_loss + beta * BCE_loss + gamma * Ranking_loss
```

## 下一步建议

1. **超参数调优**: 尝试不同的学习率、隐藏维度、损失权重
2. **增加训练轮数**: 使用更多epoch进行充分训练
3. **数据增强**: 如果有更多历史数据，可以扩展数据集
4. **特征工程**: 尝试添加更多技术指标
5. **模型集成**: 结合多个模型的预测结果

## 总结

已完成:
- ✅ 构建完整的图结构
- ✅ 修复所有数据处理问题
- ✅ 适配模型到实际数据
- ✅ 验证所有功能正常工作
- ✅ 添加完整的文档和测试

代码现在**完全可以开始训练**！
