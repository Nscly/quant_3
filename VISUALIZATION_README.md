# 📊 FinGAT 评价指标可视化系统

> **新增功能**: 完整的评价指标分析和可视化系统

---

## 🎯 快速演示

```bash
# 一键生成示例可视化图表
python visualize_metrics.py --mode demo
```

**输出**: 5张精美的可视化图表，展示模型的所有评价指标！

![Example](https://img.shields.io/badge/Status-Ready-success)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📚 文档体系

### 1️⃣ 评价指标深度分析
**文件**: [METRICS_ANALYSIS.md](./METRICS_ANALYSIS.md)  
**内容**: 
- 📖 5个核心指标的详细说明
- 🧮 数学公式和代码实现
- 📊 计算示例和解读标准
- 💡 优缺点分析和使用建议
- 🎨 可视化设计原则

**亮点**:
- MAE (平均绝对误差) - 衡量预测精度
- Accuracy (准确率) - 衡量方向判断
- MRR (平均倒数排名) - 衡量推荐质量 ⭐ 主要指标
- Precision@K (Top-K精度) - 衡量命中率
- IRR (投资回报差) - 衡量收益损失

### 2️⃣ 可视化系统
**文件**: [visualize_metrics.py](./visualize_metrics.py)  
**功能**:
- ✅ 训练过程可视化 - 4个指标的动态曲线
- ✅ 预测分析可视化 - 7个子图深度分析
- ✅ 分类任务可视化 - 混淆矩阵、ROC曲线等
- ✅ 排序任务可视化 - MRR、Precision、IRR对比
- ✅ 综合仪表盘 - 一图看全部指标

### 3️⃣ 使用指南
**文件**: [VISUALIZATION_GUIDE.md](./VISUALIZATION_GUIDE.md)  
**内容**:
- 🚀 快速开始教程
- 📈 每个图表的详细解读
- 🎨 自定义配置方法
- 🔧 与训练脚本集成
- 💡 使用建议和最佳实践

---

## 🎨 可视化图表预览

### 1. 训练曲线
```
┌─────────────┬─────────────┐
│  MAE曲线    │ Accuracy曲线 │
├─────────────┼─────────────┤
│  MRR曲线 ⭐ │ Precision曲线│
└─────────────┴─────────────┘
```
展示训练过程中4个核心指标的变化趋势

### 2. 预测分析图
```
┌──────────┬──────────┬──────────┐
│预测vs实际│ 误差分布  │ 误差箱线图│
├──────────┼──────────┼──────────┤
│  残差图   │  Q-Q图   │ 累积分布 │
├──────────┴──────────┴──────────┤
│        绝对误差分布            │
└───────────────────────────────┘
```
7个子图全方位分析回归预测性能

### 3. 分类分析图
```
┌──────────┬──────────┬──────────┐
│ 混淆矩阵  │ 指标对比  │ 类别分布 │
├──────────┼──────────┼──────────┤
│收益率分布 │ ROC曲线  │ 统计报告 │
└──────────┴──────────┴──────────┘
```
深度分析涨跌预测准确性

### 4. 排序分析图
```
┌──────────────┬──────────┬──────────┐
│ 排名对比散点图 │ MRR对比  │Precision │
├──────────────┼──────────┼──────────┤
│  收益对比    │   IRR    │ 命中饼图 │
├──────────────┴──────────┴──────────┤
│          累积收益曲线              │
└───────────────────────────────────┘
```
评估Top-K推荐质量

### 5. 综合仪表盘
```
┌──────────┬──────────┬──────────┬──────────┐
│        MAE曲线        │       MRR曲线        │
├──────────┴──────────┼──────────┴──────────┤
│   预测vs实际散点图    │      混淆矩阵        │
├─────┬─────┬─────┬───┴───┬──────┬──────────┤
│ MAE │ Acc │ MRR │Prec@5 │误差分布│Precision│
└─────┴─────┴─────┴───────┴──────┴──────────┘
```
一图汇总所有关键信息

---

## 🚀 使用方法

### 基础使用

```bash
# 1. 演示模式（使用模拟数据）
python visualize_metrics.py --mode demo

# 2. 指定输出目录
python visualize_metrics.py --mode demo --output my_visualizations

# 3. 分析模式（使用实际训练数据）
python visualize_metrics.py --mode analysis --data training_results.pkl
```

### 完整工作流

```bash
# 步骤1: 训练模型
python train.py --model CAT --epochs 20

# 步骤2: 可视化结果
python visualize_metrics.py --mode demo

# 步骤3: 查看图表
cd visualizations/
ls -lh *.png
```

---

## 📊 指标速查

### 性能标准

| 指标 | 优秀 | 良好 | 一般 | 需改进 |
|------|------|------|------|--------|
| MAE | < 0.020 | 0.020-0.030 | 0.030-0.050 | > 0.050 |
| Accuracy | > 0.65 | 0.60-0.65 | 0.55-0.60 | < 0.55 |
| MRR@5 | > 3.5 | 3.0-3.5 | 2.5-3.0 | < 2.5 |
| Precision@5 | > 0.70 | 0.60-0.70 | 0.50-0.60 | < 0.50 |

### 指标含义

**MAE (Mean Absolute Error)**
- 预测收益率与实际收益率的平均偏差
- 例: MAE=0.02 表示平均误差2%

**Accuracy**
- 正确预测涨跌方向的比例
- 例: Acc=0.65 表示65%预测正确

**MRR (Mean Reciprocal Rank)**
- 推荐质量的综合评分
- 例: MRR@5=3.5 表示推荐的5支股票排名很好

**Precision@K**
- 推荐的K支股票中真正优秀的比例
- 例: Precision@5=0.7 表示推荐5支中有3.5支确实很好

---

## 💡 典型应用场景

### 场景1: 模型调优
**问题**: 不知道调整哪个超参数？  
**解决**: 
1. 查看训练曲线，判断是否欠拟合/过拟合
2. 分析误差分布，找出系统性偏差
3. 根据指标调整：
   - MAE高 → 增加模型容量
   - Accuracy低 → 检查分类阈值
   - MRR低 → 调整gamma参数（排序损失权重）

### 场景2: 模型对比
**问题**: CG、CAT、CPool哪个更好？  
**解决**:
1. 分别训练三个模型
2. 生成可视化报告
3. 对比综合仪表盘
4. 根据业务需求选择（速度 vs 性能）

### 场景3: 生产部署评估
**问题**: 模型是否达到上线标准？  
**解决**:
1. 查看所有5个指标是否达标
2. 特别关注MRR和Precision（推荐核心）
3. 分析误差分布，评估风险
4. 检查混淆矩阵，确保FP率可接受

### 场景4: 问题诊断
**问题**: 模型表现突然下降？  
**解决**:
1. 对比历史可视化图表
2. 检查预测分布是否异常
3. 分析哪个指标下降最多
4. 针对性改进

---

## 🔧 高级定制

### 修改颜色方案

```python
# 在visualize_metrics.py中修改
self.colors = {
    'MAE': '#your_color',
    'Accuracy': '#your_color',
    'MRR': '#your_color',
    'Precision': '#your_color',
    'IRR': '#your_color',
}
```

### 添加新指标

```python
# 1. 在train.py中计算新指标
new_metric = calculate_new_metric(y_true, y_pred)

# 2. 保存到历史
metrics_history['new_metric'] = new_metric_history

# 3. 在visualize_metrics.py中添加绘图代码
def plot_new_metric(self, data):
    plt.plot(data)
    plt.title('New Metric')
    # ...
```

### 生成PDF报告

```python
# 安装reportlab
pip install reportlab

# 使用matplotlib后端
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages('report.pdf') as pdf:
    # 生成所有图表
    visualizer.plot_training_curves(...)
    pdf.savefig()
    # ...
```

---

## 📦 依赖项

### 必需
- `numpy` ✅
- `pandas` ✅
- `matplotlib` ✅
- `scikit-learn` ✅

### 可选
- `seaborn` - 更美观的图表样式
- `scipy` - Q-Q图等统计图表
- `plotly` - 交互式可视化（未来支持）

**注意**: 即使没有seaborn和scipy，脚本也能正常运行！

---

## 🎓 学习路径

### 初学者
1. 阅读 [METRICS_ANALYSIS.md](./METRICS_ANALYSIS.md) 了解指标含义
2. 运行 `python visualize_metrics.py --mode demo` 看示例
3. 阅读 [VISUALIZATION_GUIDE.md](./VISUALIZATION_GUIDE.md) 学习解读

### 进阶用户
1. 修改train.py保存训练数据
2. 使用实际数据生成可视化
3. 根据图表调优模型
4. 对比不同模型性能

### 高级用户
1. 自定义可视化样式
2. 添加新的评价指标
3. 开发交互式Dashboard
4. 集成到CI/CD流程

---

## 📖 完整文档列表

```
评价指标和可视化:
├── METRICS_ANALYSIS.md          - 指标深度分析 (⭐核心文档)
├── VISUALIZATION_GUIDE.md       - 可视化使用指南
├── VISUALIZATION_README.md      - 本文件
└── visualize_metrics.py         - 可视化脚本

训练和模型:
├── README_CN.md                 - 项目总览
├── TRAINING_GUIDE.md            - 训练指南
├── ARCHITECTURE.md              - 架构文档
├── TRAINING_ANALYSIS.md         - 代码分析
└── SUMMARY_CN.md                - 总结报告
```

---

## ❓ 常见问题

**Q: 为什么我的图表是英文的？**  
A: 系统可能没有中文字体。可以修改代码使用英文，或安装中文字体。

**Q: 图表太小看不清？**  
A: 修改 `plt.rcParams['figure.figsize']` 参数，或保存后放大查看。

**Q: 可以导出Excel报告吗？**  
A: 当前未支持，但可以用pandas将指标保存为CSV。

**Q: 如何实时监控训练？**  
A: 未来版本将支持，当前可以定期运行可视化脚本。

**Q: 能生成交互式图表吗？**  
A: 当前使用matplotlib，未来计划支持plotly交互式图表。

---

## 🌟 特色功能

### ✅ 已实现
- 5种核心评价指标的全面分析
- 5张详细的可视化图表
- 演示模式快速预览
- 无seaborn也能运行
- 完整的文档体系

### 🚧 计划中
- 实时训练监控
- 交互式Dashboard
- 模型对比功能
- PDF报告导出
- 命令行交互界面

---

## 🙏 致谢

感谢FinGAT原作者提供优秀的模型架构！

本可视化系统专为FinGAT设计，帮助用户更好地理解和优化模型性能。

---

## 📞 获取帮助

- 📖 详细说明: [VISUALIZATION_GUIDE.md](./VISUALIZATION_GUIDE.md)
- 📊 指标分析: [METRICS_ANALYSIS.md](./METRICS_ANALYSIS.md)
- 🏗️ 架构文档: [ARCHITECTURE.md](./ARCHITECTURE.md)
- 📚 项目总览: [README_CN.md](./README_CN.md)

---

**开始使用: `python visualize_metrics.py --mode demo`** 📊✨
