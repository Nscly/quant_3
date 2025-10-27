# FinGAT 可视化系统使用指南

## 📊 概述

本可视化系统为FinGAT模型提供全面的评价指标可视化功能，帮助您深入理解模型的性能表现。

**已创建文件:**
- `METRICS_ANALYSIS.md` - 评价指标深度分析文档
- `visualize_metrics.py` - 可视化脚本

---

## 🎯 评价指标概览

### 五大核心指标

| 指标 | 中文名称 | 类型 | 计算方法 | 最佳值 | 用途 |
|------|---------|------|---------|--------|------|
| **MAE** | 平均绝对误差 | 回归 | `mean(\|pred - actual\|)` | 越小越好 | 衡量预测精度 |
| **Accuracy** | 准确率 | 分类 | `correct / total` | 越大越好 | 衡量方向判断 |
| **MRR** | 平均倒数排名 | 排序 | `Σ(1/rank)` for top-K | 越大越好 | 衡量推荐质量 ⭐ |
| **Precision@K** | Top-K精度 | 推荐 | `\|pred∩true\| / K` | 越大越好 | 衡量命中率 |
| **IRR** | 投资回报差 | 收益 | `ideal - actual` | 越小越好 | 衡量收益损失 |

### 详细说明请参考
👉 **[METRICS_ANALYSIS.md](./METRICS_ANALYSIS.md)** - 包含每个指标的：
- 数学公式
- 代码实现
- 计算示例
- 解读标准
- 优缺点分析
- 可视化建议

---

## 🚀 快速开始

### 方法1: 演示模式（推荐初次使用）

```bash
python visualize_metrics.py --mode demo
```

**功能:**
- 使用模拟数据生成示例图表
- 展示所有可视化类型
- 无需实际训练数据
- 快速了解可视化效果

**输出:**
```
visualizations/
├── training_curves.png           - 训练过程曲线
├── prediction_analysis.png       - 预测分析图
├── classification_analysis.png   - 分类分析图
├── ranking_analysis.png          - 排序分析图
└── comprehensive_dashboard.png   - 综合仪表盘
```

### 方法2: 分析模式（使用实际数据）

```bash
python visualize_metrics.py --mode analysis --data <结果文件路径>
```

**注意:** 需要先修改train.py保存预测结果

---

## 📈 可视化图表详解

### 1. 训练曲线 (training_curves.png)

**内容:**
- 2×2子图布局
- 显示4个指标随epoch的变化
- 标注最佳点和最佳epoch

**包含图表:**

#### 1.1 MAE训练曲线
- **X轴**: Epoch
- **Y轴**: MAE值
- **特点**:
  - 红色虚线标注最小值
  - 红色星号标记最佳epoch
  - 理想情况：持续下降并趋于稳定

**解读:**
```
下降趋势良好 ✅
  - 说明模型在学习
  - 预测精度提升

平台期 ⚠️
  - 可能需要调整学习率
  - 或增加模型容量

上升 ❌
  - 可能过拟合
  - 需要正则化
```

#### 1.2 Accuracy训练曲线
- **范围**: 0-1
- **基线**: 0.5 (随机猜测)
- **目标**: > 0.6

#### 1.3 MRR训练曲线 ⭐
- **主要选择指标**
- 体现排序质量
- 越高越好

#### 1.4 Precision训练曲线
- **范围**: 0-1
- 推荐系统核心指标
- 与MRR相关但不同

---

### 2. 预测分析图 (prediction_analysis.png)

**布局**: 3×3网格，7个子图

#### 2.1 预测 vs 实际 散点图
- **理想**: 所有点在y=x线上
- **偏离**: 说明预测有偏差
- **相关系数**: 越接近1越好

**质量评估:**
```
相关系数 > 0.8  ✅ 优秀
         0.6-0.8  ⚠️ 良好
         < 0.6  ❌ 需改进
```

#### 2.2 误差分布直方图
- **期望**: 接近正态分布，中心在0
- **偏移**: 系统性偏差
- **长尾**: 异常预测

#### 2.3 残差图
- **检测**: 系统性偏差
- **理想**: 随机分布在y=0两侧
- **模式**: 如果有规律，说明模型未学到某些特征

#### 2.4 误差箱线图
- **展示**: 误差的统计分布
- **关注**: 离群点（outliers）

#### 2.5 Q-Q图
- **检验**: 误差是否正态分布
- **理想**: 点在对角线上
- **偏离**: 说明分布异常

#### 2.6 累积误差分布 (CDF)
- **X轴**: 绝对误差
- **Y轴**: 累积概率
- **用途**: 了解X%的样本误差在Y以内

**示例解读:**
```
80%的样本误差 < 0.02
90%的样本误差 < 0.03
95%的样本误差 < 0.05
```

#### 2.7 绝对误差分布
- 与误差分布类似
- 只关注误差大小，不关注方向

---

### 3. 分类分析图 (classification_analysis.png)

**布局**: 2×3网格，6个子图

#### 3.1 混淆矩阵热力图
```
实际\预测 |  跌(0)  |  涨(1)
----------|---------|--------
  跌(0)   |   TN    |   FP   ← 误判为涨
  涨(1)   |   FN    |   TP   ← 正确预测涨
            ↑ 误判为跌
```

**关键指标:**
- **TP (True Positive)**: 预测涨且实际涨
- **TN (True Negative)**: 预测跌且实际跌
- **FP (False Positive)**: 预测涨但实际跌 ⚠️ 损失风险
- **FN (False Negative)**: 预测跌但实际涨 ⚠️ 机会损失

#### 3.2 分类指标柱状图
- **Accuracy**: (TP+TN) / Total
- **Precision**: TP / (TP+FP)
- **Recall**: TP / (TP+FN)
- **F1-Score**: 2×Precision×Recall / (Precision+Recall)

#### 3.3 类别分布对比
- 对比实际和预测的涨跌比例
- 检查是否有预测偏向

#### 3.4 预测收益率分布（按实际标签）
- 红色：实际下跌股票的预测分布
- 绿色：实际上涨股票的预测分布
- 黑色虚线：决策边界 (0)

**理想情况:**
```
红色分布 < 0 (正确预测跌)
绿色分布 > 0 (正确预测涨)
两个分布分离度越大越好
```

#### 3.5 ROC曲线
- **AUC**: 曲线下面积
- **AUC = 0.5**: 随机猜测
- **AUC > 0.7**: 有预测能力
- **AUC > 0.8**: 预测能力强

#### 3.6 统计信息表格
- 详细的分类报告
- 包含所有关键指标
- 错误分析

---

### 4. 排序分析图 (ranking_analysis.png)

**布局**: 3×3网格，7个子图

#### 4.1 Top-20排名对比散点图
- **X轴**: 预测排名
- **Y轴**: 实际排名
- **对角线**: 完美预测
- **偏离度**: 预测误差

**解读:**
```
点在对角线上 ✅ 完美
点在对角线附近 ✅ 良好
点远离对角线 ❌ 排序错误
```

#### 4.2 MRR@K柱状图
- 对比K=5/10/20的MRR得分
- 一般K越大，MRR越高（更容易命中）

**标准:**
```
MRR@5 > 3.0  ✅ 优秀
      2.0-3.0  ⚠️ 良好
      < 2.0  ❌ 需改进
```

#### 4.3 Precision@K柱状图
- 对比不同K值的精度
- 直观显示推荐质量

**标准:**
```
Precision@5 > 0.7  ✅ 优秀
            0.6-0.7  ⚠️ 良好
            < 0.6  ❌ 需改进
```

#### 4.4 Top-K收益对比
- 绿色：理想投资组合收益
- 蓝色：实际推荐组合收益
- 差距越小越好

#### 4.5 IRR（投资回报差）
- 显示不同K值的收益损失
- 理想值：0
- 越小越好

#### 4.6 Top-K命中情况饼图
- **绿色**: 命中（预测对了）
- **橙色**: 仅预测（预测了但不是最好的）
- **红色**: 仅实际（错过了）

#### 4.7 累积收益曲线
- **绿线**: 理想累积收益
- **蓝线**: 实际推荐累积收益
- **红色区域**: 收益损失
- **垂直线**: 标记K=5/10/20位置

**用途:**
- 看不同K值下的收益差距
- 找到最优的K值

---

### 5. 综合仪表盘 (comprehensive_dashboard.png)

**布局**: 4×4网格，一览全部指标

#### 第一行：训练曲线
- 左：MAE曲线
- 右：MRR曲线

#### 第二行：核心图表
- 左：预测vs实际散点图
- 右：混淆矩阵

#### 第三行：指标卡片
```
┌─────────┬─────────┬─────────┬─────────┐
│  MAE    │ Accuracy│  MRR@5  │Precision│
│ 0.0234  │  0.6234 │  0.7543 │  0.6800 │
└─────────┴─────────┴─────────┴─────────┘
```

#### 第四行：分布和对比
- 左：误差分布直方图
- 右：不同K值的Precision柱状图

#### 右下角：统计信息
- 完整的模型评估报告
- 所有关键指标汇总

---

## 🎨 自定义输出

### 修改输出目录

```bash
python visualize_metrics.py --mode demo --output my_visualizations
```

### 修改颜色配置

编辑 `visualize_metrics.py` 中的 `colors` 字典：

```python
self.colors = {
    'MAE': '#1f77b4',      # 蓝色
    'Accuracy': '#2ca02c', # 绿色
    'MRR': '#ff7f0e',      # 橙色
    'Precision': '#9467bd',# 紫色
    'IRR': '#d62728',      # 红色
}
```

---

## 🔧 与训练脚本集成

### 修改train.py保存数据

在 `train.py` 中添加数据保存功能：

```python
# 在train函数结尾添加
def train(args):
    # ... 原有训练代码 ...
    
    # 保存训练历史
    metrics_history = {
        'epochs': list(range(1, epochs+1)),
        'mae': mae_history,
        'accuracy': acc_history,
        'mrr': mrr_history,
        'precision': precision_history
    }
    
    # 保存预测结果
    results_dict = {
        'metrics_history': metrics_history,
        'y_true': test_y,
        'y_pred': y_pred,
        'y_true_class': test_cls
    }
    
    # 保存为pickle
    with open('training_results.pkl', 'wb') as f:
        pickle.dump(results_dict, f)
    
    print("✓ 训练结果已保存到 training_results.pkl")
    
    return best_metric_IRR, best_metric_MRR, best_results_IRR, best_results_MRR
```

### 使用保存的数据可视化

```python
# 在visualize_metrics.py中添加加载功能
import pickle

def load_training_results(filepath):
    """加载训练结果"""
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    return results

# 使用
if args.mode == 'analysis':
    results = load_training_results(args.data)
    visualizer.generate_report(
        results['metrics_history'],
        results['y_true'],
        results['y_pred'],
        results['y_true_class']
    )
```

---

## 📊 实际使用流程

### 完整工作流

```bash
# 1. 训练模型（会保存结果）
python train.py --model CAT --epochs 20

# 2. 可视化结果
python visualize_metrics.py --mode analysis --data training_results.pkl

# 3. 查看生成的图表
cd visualizations/
ls -lh
```

### 对比不同模型

```bash
# 训练多个模型
python train.py --model CG --epochs 20
mv training_results.pkl results_CG.pkl

python train.py --model CAT --epochs 20
mv training_results.pkl results_CAT.pkl

python train.py --model CPool --epochs 20
mv training_results.pkl results_CPool.pkl

# 对比可视化（需要实现compare模式）
python visualize_metrics.py --mode compare \
    --data results_CG.pkl results_CAT.pkl results_CPool.pkl
```

---

## 💡 使用建议

### 训练阶段
1. **每10个epoch**查看一次训练曲线
2. 关注是否**过拟合**（训练loss下降但验证loss上升）
3. 根据曲线调整学习率和正则化参数

### 评估阶段
1. **首先查看**综合仪表盘，获得整体印象
2. **深入分析**各个指标的详细图表
3. **重点关注** MRR和Precision（推荐任务核心）

### 调优阶段
1. **MAE高**: 增加模型容量，降低学习率
2. **Accuracy低**: 检查分类阈值，平衡样本
3. **MRR低**: 优化排序损失权重（gamma参数）
4. **Precision低**: 增加训练数据，改进特征工程

---

## 🎯 指标解读速查表

### MAE (Mean Absolute Error)
```
< 0.020  ✅ 优秀 - 平均误差2%
0.020-0.030  ⚠️ 良好 - 平均误差2-3%
0.030-0.050  ⚠️ 一般 - 平均误差3-5%
> 0.050  ❌ 较差 - 平均误差>5%
```

### Accuracy
```
> 0.65  ✅ 优秀
0.60-0.65  ⚠️ 良好
0.55-0.60  ⚠️ 一般
0.50-0.55  ❌ 接近随机
< 0.50  ❌ 低于随机
```

### MRR@5
```
> 3.5  ✅ 优秀
3.0-3.5  ⚠️ 良好
2.5-3.0  ⚠️ 一般
< 2.5  ❌ 需改进
```

### Precision@5
```
> 0.70  ✅ 优秀 - 70%命中
0.60-0.70  ⚠️ 良好 - 60-70%命中
0.50-0.60  ⚠️ 一般 - 50-60%命中
< 0.50  ❌ 需改进 - 50%以下
```

---

## 📝 常见问题

### Q1: 生成的图表中文显示乱码？
**A:** 系统可能缺少中文字体。修改 `visualize_metrics.py`:
```python
plt.rcParams['font.sans-serif'] = ['Arial']  # 使用英文
```

### Q2: 图表太大/太小？
**A:** 修改图表大小：
```python
plt.rcParams['figure.figsize'] = (20, 14)  # 调整尺寸
```

### Q3: 想要高分辨率图片？
**A:** 修改DPI设置：
```python
plt.savefig(save_path, dpi=600)  # 默认300，可提高到600
```

### Q4: 如何只生成某一个图表？
**A:** 调用单个函数：
```python
visualizer = MetricsVisualizer()
visualizer.plot_training_curves(metrics_history)
```

### Q5: 颜色看不清楚？
**A:** 修改颜色方案或使用更high contrast的配色

---

## 🔍 进阶功能（待实现）

### 交互式Dashboard
```bash
python visualize_metrics.py --mode dashboard
# 启动Web界面，可交互查看指标
```

### 实时监控
```bash
python visualize_metrics.py --mode realtime --logfile training.log
# 实时读取训练日志并更新图表
```

### 对比模式
```bash
python visualize_metrics.py --mode compare \
    --models CG CAT CPool \
    --data results_*.pkl
# 并排对比不同模型的性能
```

---

## 📚 相关文档

- **[METRICS_ANALYSIS.md](./METRICS_ANALYSIS.md)** - 评价指标深度分析
- **[TRAINING_GUIDE.md](./TRAINING_GUIDE.md)** - 训练流程指南
- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - 模型架构文档
- **[README_CN.md](./README_CN.md)** - 项目总览

---

## 🎓 最佳实践总结

### 1. 训练前
- ✅ 了解每个指标的含义
- ✅ 设定性能目标
- ✅ 准备对比基线

### 2. 训练中
- ✅ 定期生成可视化
- ✅ 关注趋势而非绝对值
- ✅ 及时调整超参数

### 3. 训练后
- ✅ 生成完整报告
- ✅ 深入分析每个指标
- ✅ 记录最佳配置

### 4. 部署前
- ✅ 在测试集上验证
- ✅ 对比历史最佳模型
- ✅ 评估业务指标

---

**祝您使用愉快！如有问题请查阅[METRICS_ANALYSIS.md](./METRICS_ANALYSIS.md)或源代码注释。📊**
