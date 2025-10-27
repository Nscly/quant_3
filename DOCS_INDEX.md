# 📚 FinGAT 文档索引

欢迎！这里是FinGAT项目的文档导航页面。根据您的需求选择相应的文档：

## 🚀 快速开始

**我是新手，想快速了解项目:**
- 👉 阅读 **[README_CN.md](./README_CN.md)** - 5分钟快速入门

**我想立即开始训练:**
```bash
# 方法1: 使用脚本（最简单）
./quick_train.sh

# 方法2: 手动执行
source .venv/bin/activate
python prepare_data.py
python train.py --model CAT --device cpu --epochs 10
```

## 📖 文档导航

### 1️⃣ 入门级文档

#### [README_CN.md](./README_CN.md) 📱
**适合人群:** 初次使用者
**内容:**
- 项目简介和特性
- 快速开始指南
- 数据集说明
- 训练步骤详解
- 常见问题解答
- 参数说明表格

**阅读时间:** 10-15分钟

---

### 2️⃣ 训练指南

#### [TRAINING_GUIDE.md](./TRAINING_GUIDE.md) 🎓
**适合人群:** 需要深入了解训练流程的用户
**内容:**
- 完整训练流程详解
- 数据处理步骤
- 三种模型对比（CG/CAT/CPool）
- 超参数调优建议
- 损失函数说明
- 评估指标解读
- 最佳实践建议
- 性能优化技巧

**阅读时间:** 30-40分钟

---

### 3️⃣ 架构文档

#### [ARCHITECTURE.md](./ARCHITECTURE.md) 🏗️
**适合人群:** 需要理解技术细节的开发者和研究者
**内容:**
- 系统整体架构图
- 数据流详解
- 三种模型架构对比
- 核心组件说明（AttentionBlock, SequenceEncoder等）
- 损失函数数学公式
- 双层图结构设计
- 关键技术点分析
- 超参数影响分析
- 理论基础

**阅读时间:** 45-60分钟

---

### 4️⃣ 综合分析

#### [TRAINING_ANALYSIS.md](./TRAINING_ANALYSIS.md) 📊
**适合人群:** 项目管理者、技术评审者
**内容:**
- 执行摘要
- 项目概述
- 代码架构分析
- 实际测试结果
- 训练建议和最佳实践
- 模型选择指南
- 常见问题和解决方案
- 性能基准参考
- 技术要点总结
- 下一步建议

**阅读时间:** 30-40分钟

---

### 5️⃣ 配置参考

#### [train_configs.json](./train_configs.json) ⚙️
**适合人群:** 需要快速查找配置的用户
**内容:**
- 各种训练场景的配置示例
- 超参数搜索空间
- 命令行示例
- 评估指标说明
- 硬件需求参考

**查阅时间:** 5分钟

---

## 🛠️ 脚本和工具

### 数据准备脚本
**[prepare_data.py](./prepare_data.py)**
```bash
python prepare_data.py
```
- 自动读取股票数据
- 特征工程
- 生成滑动窗口
- 创建图边矩阵
- 保存为pickle格式

### 快速训练脚本
**[quick_train.sh](./quick_train.sh)**
```bash
./quick_train.sh [模型] [设备] [轮数]
# 例如: ./quick_train.sh CAT cuda:0 20
```
- 自动检查环境
- 自动准备数据
- 一键开始训练

---

## 📋 推荐阅读路径

### 🎯 路径1: 快速上手（30分钟）
```
1. README_CN.md (快速浏览)
2. 运行 quick_train.sh
3. 观察训练输出
4. 查阅 train_configs.json 了解参数
```

### 🎓 路径2: 深入学习（2小时）
```
1. README_CN.md (完整阅读)
2. TRAINING_GUIDE.md (重点章节)
3. 运行 prepare_data.py 理解数据流程
4. 运行 train.py 实验不同参数
5. 查阅 ARCHITECTURE.md 理解技术细节
```

### 🔬 路径3: 研究开发（半天）
```
1. TRAINING_ANALYSIS.md (全文)
2. ARCHITECTURE.md (全文)
3. TRAINING_GUIDE.md (全文)
4. 阅读源代码 (train.py, model/graph_pool.py)
5. 实验和修改
```

---

## 🔍 快速查找

### 我想知道...

**如何开始训练？**
→ [README_CN.md - 快速开始](./README_CN.md#-快速开始)

**有哪些参数可以调？**
→ [README_CN.md - 参数调优](./README_CN.md#步骤4-参数调优)
→ [train_configs.json](./train_configs.json)

**模型架构是什么样的？**
→ [ARCHITECTURE.md - 模型架构对比](./ARCHITECTURE.md#-模型架构对比)

**如何解读训练输出？**
→ [README_CN.md - 训练输出解读](./README_CN.md#-训练输出解读)
→ [TRAINING_GUIDE.md - 步骤3](./TRAINING_GUIDE.md#步骤3-训练过程)

**遇到报错怎么办？**
→ [README_CN.md - 常见问题](./README_CN.md#-常见问题)
→ [TRAINING_ANALYSIS.md - 常见问题和解决方案](./TRAINING_ANALYSIS.md#-常见问题和解决方案)

**如何优化性能？**
→ [TRAINING_GUIDE.md - 性能优化](./TRAINING_GUIDE.md#-性能优化)
→ [ARCHITECTURE.md - 超参数影响分析](./ARCHITECTURE.md#-超参数影响分析)

**三种模型如何选择？**
→ [TRAINING_GUIDE.md - 模型架构](./TRAINING_GUIDE.md#2-模型定义模块-modelgraph_poolpy)
→ [TRAINING_ANALYSIS.md - 模型选择指南](./TRAINING_ANALYSIS.md#-模型选择指南)

**数据是如何处理的？**
→ [TRAINING_GUIDE.md - 步骤1](./TRAINING_GUIDE.md#步骤1-准备数据)
→ [ARCHITECTURE.md - 数据流详解](./ARCHITECTURE.md#-数据流详解)

**损失函数怎么设计的？**
→ [ARCHITECTURE.md - 损失函数设计](./ARCHITECTURE.md#-损失函数设计)

**评估指标是什么意思？**
→ [README_CN.md - 指标解释](./README_CN.md#指标解释)
→ [TRAINING_GUIDE.md - 评估指标](./TRAINING_GUIDE.md#评估指标)

---

## 📚 源代码导航

### 核心文件

**数据处理:**
- `clean_data.py` - 原始数据清洗（需要修改）
- `prepare_data.py` - 自动化数据准备（推荐使用）

**模型定义:**
- `model/graph_pool.py` - 三种模型实现
  - `CategoricalGraph` (CG)
  - `CategoricalGraphAtt` (CAT)
  - `CategoricalGraphPool` (CPool)

**训练和评估:**
- `train.py` - 主训练脚本
- `parse_arg.py` - 命令行参数定义

**辅助脚本:**
- `quick_train.sh` - 快速训练bash脚本

**数据:**
- `stock_data/*.csv` - 原始股票数据
- `crypto_miners_category.csv` - 类别标签

**生成文件:**
- `Taiwan_model_data_10_best.pickle` - 训练/测试数据
- `*.npy` - 图边矩阵文件

---

## 💡 使用建议

### 对于初学者
1. 先运行一次完整流程（使用quick_train.sh）
2. 观察输出，理解训练过程
3. 阅读README_CN.md了解参数含义
4. 尝试修改参数，观察效果

### 对于开发者
1. 阅读ARCHITECTURE.md理解设计思路
2. 阅读源码，特别是model/graph_pool.py
3. 尝试修改模型结构
4. 实验不同的训练策略

### 对于研究者
1. 完整阅读所有文档
2. 理解理论基础和创新点
3. 尝试改进模型架构
4. 进行消融实验

---

## 🆘 获取帮助

### 步骤1: 查阅文档
- 先查看 [README_CN.md - 常见问题](./README_CN.md#-常见问题)
- 如果是技术问题，查看 [TRAINING_ANALYSIS.md - 常见问题](./TRAINING_ANALYSIS.md#-常见问题和解决方案)

### 步骤2: 查看日志
- 检查训练输出中的错误信息
- 查看是否有数据或模型相关的warning

### 步骤3: 检查环境
```bash
# 检查Python环境
python --version

# 检查依赖包
pip list | grep -E "torch|pandas|numpy"

# 检查数据文件
ls -lh *.pickle *.npy
```

---

## 📊 文档统计

| 文档 | 字数 | 代码示例 | 适合人群 | 推荐度 |
|------|------|----------|----------|--------|
| README_CN.md | ~8K | 30+ | 入门 | ⭐⭐⭐⭐⭐ |
| TRAINING_GUIDE.md | ~12K | 40+ | 进阶 | ⭐⭐⭐⭐ |
| ARCHITECTURE.md | ~15K | 50+ | 高级 | ⭐⭐⭐⭐ |
| TRAINING_ANALYSIS.md | ~13K | 35+ | 全面 | ⭐⭐⭐⭐⭐ |
| train_configs.json | ~2K | 15+ | 参考 | ⭐⭐⭐ |

---

## 🎯 目标读者

### 👨‍💼 项目管理者
- 阅读: TRAINING_ANALYSIS.md
- 了解: 项目状态、性能基准、资源需求
- 时间: 30分钟

### 👨‍💻 开发工程师
- 阅读: README_CN.md + ARCHITECTURE.md
- 了解: 如何使用、如何修改、技术细节
- 时间: 2小时

### 👨‍🔬 算法研究员
- 阅读: 所有文档 + 源代码
- 了解: 理论基础、实现细节、优化方向
- 时间: 半天

### 👨‍🎓 学生学习者
- 阅读: README_CN.md + TRAINING_GUIDE.md
- 了解: 基本概念、训练流程、实践经验
- 时间: 3小时

---

## 📅 文档更新

- **初始版本**: 2024年
- **最后更新**: 2024年
- **维护状态**: 活跃
- **反馈渠道**: 欢迎提出改进建议

---

## ✅ 检查清单

**在开始训练前，请确保:**
- [ ] 已阅读 README_CN.md
- [ ] 已准备好数据文件
- [ ] 已安装所需依赖
- [ ] 已选择合适的模型和参数
- [ ] 已了解预期的训练时间和输出

**训练过程中:**
- [ ] 观察损失是否下降
- [ ] 检查评估指标是否合理
- [ ] 保存重要的训练日志
- [ ] 记录最佳配置和结果

**训练完成后:**
- [ ] 保存最佳模型
- [ ] 记录性能指标
- [ ] 分析实验结果
- [ ] 规划下一步优化方向

---

**祝您使用愉快！如有任何问题，请查阅相应文档。📚**

**快速链接:**
- [开始训练](./README_CN.md#-快速开始)
- [参数配置](./train_configs.json)
- [问题排查](./README_CN.md#-常见问题)
- [性能优化](./TRAINING_GUIDE.md#-性能优化)
