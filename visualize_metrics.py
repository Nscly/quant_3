"""
FinGAT 评价指标可视化系统

功能:
1. 训练过程可视化 - 实时监控指标变化
2. 最终结果可视化 - 详细分析各项指标
3. 对比分析可视化 - 不同模型/参数对比
4. 分布分析可视化 - 误差和预测分布
5. 交互式Dashboard - 综合展示所有指标

使用方法:
python visualize_metrics.py --mode <模式> [其他参数]

模式选项:
  realtime  - 实时监控训练过程
  analysis  - 分析训练日志
  compare   - 对比多个模型
  dashboard - 启动交互式仪表盘
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import json
import pickle
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 尝试导入seaborn（可选）
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("⚠️  Seaborn未安装，将使用matplotlib的默认样式")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
if HAS_SEABORN:
    sns.set_style("whitegrid")
else:
    plt.style.use('default')
    
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10


class MetricsVisualizer:
    """评价指标可视化类"""
    
    def __init__(self, output_dir="visualizations"):
        """初始化可视化器"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 颜色配置
        self.colors = {
            'MAE': '#1f77b4',      # 蓝色 - 理性、精确
            'Accuracy': '#2ca02c', # 绿色 - 正确、成功
            'MRR': '#ff7f0e',      # 橙色 - 重要、关注
            'Precision': '#9467bd',# 紫色 - 高端、精英
            'IRR': '#d62728',      # 红色 - 收益、损失
            'train': '#3498db',
            'test': '#e74c3c',
            'best': '#f39c12'
        }
        
        print(f"✓ 可视化器初始化完成，输出目录: {self.output_dir}")
    
    def plot_training_curves(self, metrics_history, save_name="training_curves.png"):
        """
        绘制训练过程曲线
        
        参数:
            metrics_history: dict, 格式: {
                'epochs': [1, 2, 3, ...],
                'mae': [0.025, 0.023, ...],
                'accuracy': [0.62, 0.64, ...],
                'mrr': [0.72, 0.75, ...],
                'precision': [0.65, 0.68, ...]
            }
        """
        print("\n[1] 绘制训练曲线...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('FinGAT 训练过程 - 评价指标变化曲线', fontsize=16, fontweight='bold')
        
        epochs = metrics_history.get('epochs', range(len(metrics_history.get('mae', []))))
        
        # 1. MAE曲线
        ax = axes[0, 0]
        mae = metrics_history.get('mae', [])
        if mae:
            ax.plot(epochs, mae, marker='o', linewidth=2, 
                   color=self.colors['MAE'], label='MAE')
            ax.axhline(y=min(mae), color='red', linestyle='--', 
                      alpha=0.5, label=f'最小值: {min(mae):.4f}')
            best_epoch = epochs[mae.index(min(mae))]
            ax.scatter([best_epoch], [min(mae)], color='red', 
                      s=200, marker='*', zorder=5, label=f'最佳 (Epoch {best_epoch})')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('MAE', fontsize=12)
        ax.set_title('MAE - 平均绝对误差 (越小越好)', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Accuracy曲线
        ax = axes[0, 1]
        accuracy = metrics_history.get('accuracy', [])
        if accuracy:
            ax.plot(epochs, accuracy, marker='s', linewidth=2, 
                   color=self.colors['Accuracy'], label='Accuracy')
            ax.axhline(y=max(accuracy), color='red', linestyle='--', 
                      alpha=0.5, label=f'最大值: {max(accuracy):.4f}')
            best_epoch = epochs[accuracy.index(max(accuracy))]
            ax.scatter([best_epoch], [max(accuracy)], color='red', 
                      s=200, marker='*', zorder=5, label=f'最佳 (Epoch {best_epoch})')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Accuracy - 涨跌预测准确率 (越大越好)', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # 3. MRR曲线
        ax = axes[1, 0]
        mrr = metrics_history.get('mrr', [])
        if mrr:
            ax.plot(epochs, mrr, marker='^', linewidth=2, 
                   color=self.colors['MRR'], label='MRR')
            ax.axhline(y=max(mrr), color='red', linestyle='--', 
                      alpha=0.5, label=f'最大值: {max(mrr):.4f}')
            best_epoch = epochs[mrr.index(max(mrr))]
            ax.scatter([best_epoch], [max(mrr)], color='red', 
                      s=200, marker='*', zorder=5, label=f'最佳 (Epoch {best_epoch})')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('MRR', fontsize=12)
        ax.set_title('MRR - 平均倒数排名 (越大越好) ⭐主要指标', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Precision曲线
        ax = axes[1, 1]
        precision = metrics_history.get('precision', [])
        if precision:
            ax.plot(epochs, precision, marker='D', linewidth=2, 
                   color=self.colors['Precision'], label='Precision@5')
            ax.axhline(y=max(precision), color='red', linestyle='--', 
                      alpha=0.5, label=f'最大值: {max(precision):.4f}')
            best_epoch = epochs[precision.index(max(precision))]
            ax.scatter([best_epoch], [max(precision)], color='red', 
                      s=200, marker='*', zorder=5, label=f'最佳 (Epoch {best_epoch})')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Precision@5', fontsize=12)
        ax.set_title('Precision@5 - Top-5推荐精度 (越大越好)', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 训练曲线已保存: {save_path}")
        plt.close()
    
    def plot_prediction_analysis(self, y_true, y_pred, save_name="prediction_analysis.png"):
        """
        绘制预测分析图
        
        参数:
            y_true: array, 实际收益率
            y_pred: array, 预测收益率
        """
        print("\n[2] 绘制预测分析图...")
        
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig)
        fig.suptitle('FinGAT 预测分析 - 回归任务', fontsize=16, fontweight='bold')
        
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        errors = y_pred - y_true
        
        # 1. 预测 vs 实际 散点图
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.scatter(y_true, y_pred, alpha=0.5, s=30, color=self.colors['MAE'])
        
        # 添加理想线 y=x
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='理想预测线 (y=x)')
        
        # 计算相关系数
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        mae = np.mean(np.abs(errors))
        
        ax1.set_xlabel('实际收益率', fontsize=12)
        ax1.set_ylabel('预测收益率', fontsize=12)
        ax1.set_title(f'预测 vs 实际 (相关系数: {correlation:.4f}, MAE: {mae:.4f})', 
                     fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 误差分布直方图
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.hist(errors, bins=50, color=self.colors['MAE'], alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零误差')
        ax2.axvline(x=np.mean(errors), color='green', linestyle='--', 
                   linewidth=2, label=f'平均误差: {np.mean(errors):.4f}')
        ax2.set_xlabel('预测误差', fontsize=12)
        ax2.set_ylabel('频数', fontsize=12)
        ax2.set_title('误差分布', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 残差图
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.scatter(y_pred, errors, alpha=0.5, s=30, color=self.colors['MAE'])
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('预测值', fontsize=12)
        ax3.set_ylabel('残差 (预测 - 实际)', fontsize=12)
        ax3.set_title('残差图 (检查系统性偏差)', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. 误差箱线图
        ax4 = fig.add_subplot(gs[1, 2])
        box_data = [errors]
        bp = ax4.boxplot(box_data, labels=['误差'], patch_artist=True)
        bp['boxes'][0].set_facecolor(self.colors['MAE'])
        ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax4.set_ylabel('误差', fontsize=12)
        ax4.set_title('误差箱线图', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = f"""
        样本数: {len(y_true)}
        MAE: {mae:.4f}
        RMSE: {np.sqrt(np.mean(errors**2)):.4f}
        相关系数: {correlation:.4f}
        
        误差统计:
        均值: {np.mean(errors):.4f}
        中位数: {np.median(errors):.4f}
        标准差: {np.std(errors):.4f}
        最小值: {np.min(errors):.4f}
        最大值: {np.max(errors):.4f}
        """
        ax4.text(1.5, np.median(errors), stats_text, 
                fontsize=9, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 5. Q-Q图
        ax5 = fig.add_subplot(gs[2, 0])
        from scipy import stats as sp_stats
        sp_stats.probplot(errors, dist="norm", plot=ax5)
        ax5.set_title('Q-Q图 (检查正态性)', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. 累积误差分布
        ax6 = fig.add_subplot(gs[2, 1])
        sorted_errors = np.sort(np.abs(errors))
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax6.plot(sorted_errors, cumulative, linewidth=2, color=self.colors['MAE'])
        ax6.axvline(x=mae, color='red', linestyle='--', 
                   linewidth=2, label=f'MAE: {mae:.4f}')
        ax6.set_xlabel('绝对误差', fontsize=12)
        ax6.set_ylabel('累积概率', fontsize=12)
        ax6.set_title('累积误差分布 (CDF)', fontsize=13, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. 绝对误差直方图
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.hist(np.abs(errors), bins=50, color=self.colors['MAE'], 
                alpha=0.7, edgecolor='black')
        ax7.axvline(x=mae, color='red', linestyle='--', 
                   linewidth=2, label=f'MAE: {mae:.4f}')
        ax7.set_xlabel('绝对误差', fontsize=12)
        ax7.set_ylabel('频数', fontsize=12)
        ax7.set_title('绝对误差分布', fontsize=13, fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 预测分析图已保存: {save_path}")
        plt.close()
    
    def plot_classification_analysis(self, y_true_class, y_pred_return, 
                                     save_name="classification_analysis.png"):
        """
        绘制分类任务分析图
        
        参数:
            y_true_class: array, 实际涨跌标签 (0/1)
            y_pred_return: array, 预测收益率 (用于判断涨跌)
        """
        print("\n[3] 绘制分类分析图...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('FinGAT 分类分析 - 涨跌预测', fontsize=16, fontweight='bold')
        
        y_true_class = np.array(y_true_class).flatten()
        y_pred_return = np.array(y_pred_return).flatten()
        y_pred_class = (y_pred_return > 0).astype(int)  # 转换为涨跌标签
        
        # 计算混淆矩阵
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_true_class, y_pred_class)
        
        # 1. 混淆矩阵热力图
        ax = axes[0, 0]
        if HAS_SEABORN:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['跌(0)', '涨(1)'],
                       yticklabels=['跌(0)', '涨(1)'])
        else:
            # 使用matplotlib绘制混淆矩阵
            im = ax.imshow(cm, cmap='Blues', aspect='auto')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['跌(0)', '涨(1)'])
            ax.set_yticklabels(['跌(0)', '涨(1)'])
            # 添加数值标注
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center', 
                           color='white' if cm[i, j] > cm.max()/2 else 'black',
                           fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax)
        ax.set_xlabel('预测标签', fontsize=12)
        ax.set_ylabel('实际标签', fontsize=12)
        ax.set_title('混淆矩阵', fontsize=13, fontweight='bold')
        
        # 2. 准确率统计
        ax = axes[0, 1]
        accuracy = (y_true_class == y_pred_class).mean()
        tp = cm[1, 1]  # True Positive
        tn = cm[0, 0]  # True Negative
        fp = cm[0, 1]  # False Positive
        fn = cm[1, 0]  # False Negative
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [accuracy, precision, recall, f1]
        colors_bar = [self.colors['Accuracy']] * len(metrics)
        
        bars = ax.barh(metrics, values, color=colors_bar, alpha=0.7)
        ax.set_xlim([0, 1])
        ax.set_xlabel('分数', fontsize=12)
        ax.set_title('分类指标', fontsize=13, fontweight='bold')
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 0.02, i, f'{val:.4f}', 
                   va='center', fontsize=11, fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='x')
        
        # 3. 类别分布对比
        ax = axes[0, 2]
        categories = ['跌(0)', '涨(1)']
        true_counts = [np.sum(y_true_class == 0), np.sum(y_true_class == 1)]
        pred_counts = [np.sum(y_pred_class == 0), np.sum(y_pred_class == 1)]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, true_counts, width, label='实际', 
              color=self.colors['test'], alpha=0.7)
        ax.bar(x + width/2, pred_counts, width, label='预测', 
              color=self.colors['train'], alpha=0.7)
        
        ax.set_xlabel('类别', fontsize=12)
        ax.set_ylabel('样本数', fontsize=12)
        ax.set_title('类别分布对比', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. 预测收益率分布 (按实际标签分组)
        ax = axes[1, 0]
        down_returns = y_pred_return[y_true_class == 0]
        up_returns = y_pred_return[y_true_class == 1]
        
        ax.hist(down_returns, bins=30, alpha=0.6, label='实际下跌股票', 
               color='red', edgecolor='black')
        ax.hist(up_returns, bins=30, alpha=0.6, label='实际上涨股票', 
               color='green', edgecolor='black')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='决策边界')
        
        ax.set_xlabel('预测收益率', fontsize=12)
        ax.set_ylabel('频数', fontsize=12)
        ax.set_title('预测收益率分布 (按实际标签)', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. ROC曲线 (如果可以计算概率)
        ax = axes[1, 1]
        from sklearn.metrics import roc_curve, auc
        
        # 使用预测收益率作为概率分数
        fpr, tpr, thresholds = roc_curve(y_true_class, y_pred_return)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=self.colors['Accuracy'], linewidth=2,
               label=f'ROC曲线 (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='随机猜测')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC曲线', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. 统计信息表格
        ax = axes[1, 2]
        ax.axis('off')
        
        # 创建详细的分类报告
        stats_text = f"""
        分类性能报告
        {'='*40}
        
        总样本数: {len(y_true_class)}
        
        混淆矩阵:
          TN={tn:>6d}  FP={fp:>6d}
          FN={fn:>6d}  TP={tp:>6d}
        
        性能指标:
          Accuracy:  {accuracy:.4f}
          Precision: {precision:.4f}
          Recall:    {recall:.4f}
          F1-Score:  {f1:.4f}
          AUC:       {roc_auc:.4f}
        
        类别统计:
          实际下跌: {true_counts[0]} ({true_counts[0]/len(y_true_class)*100:.1f}%)
          实际上涨: {true_counts[1]} ({true_counts[1]/len(y_true_class)*100:.1f}%)
          
          预测下跌: {pred_counts[0]} ({pred_counts[0]/len(y_pred_class)*100:.1f}%)
          预测上涨: {pred_counts[1]} ({pred_counts[1]/len(y_pred_class)*100:.1f}%)
        
        错误分析:
          误判为涨: {fp} ({fp/len(y_true_class)*100:.1f}%)
          误判为跌: {fn} ({fn/len(y_true_class)*100:.1f}%)
        """
        
        ax.text(0.1, 0.5, stats_text, fontsize=10, 
               verticalalignment='center', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 分类分析图已保存: {save_path}")
        plt.close()
    
    def plot_ranking_analysis(self, y_true, y_pred, k_values=[5, 10, 20],
                             save_name="ranking_analysis.png"):
        """
        绘制排序任务分析图
        
        参数:
            y_true: array, 实际收益率
            y_pred: array, 预测收益率
            k_values: list, 要分析的K值列表
        """
        print("\n[4] 绘制排序分析图...")
        
        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(3, 3, figure=fig)
        fig.suptitle('FinGAT 排序分析 - Top-K推荐', fontsize=16, fontweight='bold')
        
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # 创建DataFrame方便排序
        df = pd.DataFrame({
            'true': y_true,
            'pred': y_pred,
            'stock_id': range(len(y_true))
        })
        
        # 按预测排序
        df_pred_sorted = df.sort_values('pred', ascending=False).reset_index(drop=True)
        df_pred_sorted['pred_rank'] = df_pred_sorted.index + 1
        
        # 按实际排序
        df_true_sorted = df.sort_values('true', ascending=False).reset_index(drop=True)
        df_true_sorted['true_rank'] = df_true_sorted.index + 1
        
        # 1. 排名对比热力图
        ax1 = fig.add_subplot(gs[0, :2])
        
        # 创建排名对比矩阵 (只显示Top-20)
        top_n = min(20, len(y_true))
        rank_matrix = np.zeros((top_n, 2))
        
        for i in range(top_n):
            stock_id = df_pred_sorted.iloc[i]['stock_id']
            true_rank = df_true_sorted[df_true_sorted['stock_id'] == stock_id].index[0] + 1
            rank_matrix[i, 0] = i + 1  # 预测排名
            rank_matrix[i, 1] = true_rank  # 实际排名
        
        # 绘制散点图显示排名对比
        ax1.scatter(rank_matrix[:, 0], rank_matrix[:, 1], s=100, 
                   c=self.colors['MRR'], alpha=0.6)
        ax1.plot([1, top_n], [1, top_n], 'r--', linewidth=2, label='完美预测线')
        
        for i in range(min(10, top_n)):  # 标注前10个
            ax1.annotate(f'{i+1}', (rank_matrix[i, 0], rank_matrix[i, 1]),
                        fontsize=8, ha='center', va='center',
                        bbox=dict(boxstyle='circle', facecolor='white', alpha=0.7))
        
        ax1.set_xlabel('预测排名', fontsize=12)
        ax1.set_ylabel('实际排名', fontsize=12)
        ax1.set_title(f'Top-{top_n} 排名对比 (越接近对角线越好)', 
                     fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, top_n+1])
        ax1.set_ylim([0, top_n+1])
        
        # 2. MRR@K 对比
        ax2 = fig.add_subplot(gs[0, 2])
        
        mrr_scores = []
        for k in k_values:
            # 计算MRR@K
            top_k_true = df_true_sorted.head(k)['stock_id'].tolist()
            mrr_sum = 0
            for stock_id in top_k_true:
                pred_rank = df_pred_sorted[df_pred_sorted['stock_id'] == stock_id].index[0] + 1
                mrr_sum += 1 / pred_rank
            mrr_scores.append(mrr_sum)
        
        bars = ax2.bar(range(len(k_values)), mrr_scores, 
                      color=self.colors['MRR'], alpha=0.7)
        ax2.set_xticks(range(len(k_values)))
        ax2.set_xticklabels([f'K={k}' for k in k_values])
        ax2.set_ylabel('MRR Score', fontsize=12)
        ax2.set_title('不同K值的MRR得分', fontsize=13, fontweight='bold')
        
        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, mrr_scores)):
            ax2.text(i, score + 0.05, f'{score:.2f}', 
                    ha='center', fontsize=11, fontweight='bold')
        
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Precision@K 对比
        ax3 = fig.add_subplot(gs[1, 0])
        
        precision_scores = []
        for k in k_values:
            # 计算Precision@K
            top_k_pred = set(df_pred_sorted.head(k)['stock_id'])
            top_k_true = set(df_true_sorted.head(k)['stock_id'])
            precision = len(top_k_pred & top_k_true) / k
            precision_scores.append(precision)
        
        bars = ax3.bar(range(len(k_values)), precision_scores, 
                      color=self.colors['Precision'], alpha=0.7)
        ax3.set_xticks(range(len(k_values)))
        ax3.set_xticklabels([f'K={k}' for k in k_values])
        ax3.set_ylabel('Precision Score', fontsize=12)
        ax3.set_title('不同K值的Precision得分', fontsize=13, fontweight='bold')
        ax3.set_ylim([0, 1])
        
        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, precision_scores)):
            ax3.text(i, score + 0.02, f'{score:.3f}', 
                    ha='center', fontsize=11, fontweight='bold')
        
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Top-K收益对比 (IRR分析)
        ax4 = fig.add_subplot(gs[1, 1])
        
        pred_returns = []
        true_returns = []
        irr_scores = []
        
        for k in k_values:
            pred_ret = df_pred_sorted.head(k)['true'].sum()
            true_ret = df_true_sorted.head(k)['true'].sum()
            pred_returns.append(pred_ret)
            true_returns.append(true_ret)
            irr_scores.append(true_ret - pred_ret)
        
        x = np.arange(len(k_values))
        width = 0.35
        
        ax4.bar(x - width/2, true_returns, width, label='理想收益', 
               color='green', alpha=0.7)
        ax4.bar(x + width/2, pred_returns, width, label='实际推荐收益', 
               color='blue', alpha=0.7)
        
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'K={k}' for k in k_values])
        ax4.set_ylabel('累积收益率', fontsize=12)
        ax4.set_title('Top-K组合收益对比', fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. IRR (收益损失)
        ax5 = fig.add_subplot(gs[1, 2])
        
        bars = ax5.bar(range(len(k_values)), irr_scores, 
                      color=self.colors['IRR'], alpha=0.7)
        ax5.axhline(y=0, color='black', linestyle='--', linewidth=2)
        ax5.set_xticks(range(len(k_values)))
        ax5.set_xticklabels([f'K={k}' for k in k_values])
        ax5.set_ylabel('IRR (收益损失)', fontsize=12)
        ax5.set_title('Investment Return Rate - 越小越好', fontsize=13, fontweight='bold')
        
        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, irr_scores)):
            y_pos = score + 0.002 if score > 0 else score - 0.002
            ax5.text(i, y_pos, f'{score:.4f}', 
                    ha='center', fontsize=11, fontweight='bold')
        
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Top-K命中情况可视化 (Venn-like diagram)
        ax6 = fig.add_subplot(gs[2, 0])
        
        k = k_values[0]  # 使用第一个K值
        top_k_pred_set = set(df_pred_sorted.head(k)['stock_id'])
        top_k_true_set = set(df_true_sorted.head(k)['stock_id'])
        
        intersection = top_k_pred_set & top_k_true_set
        only_pred = top_k_pred_set - top_k_true_set
        only_true = top_k_true_set - top_k_pred_set
        
        sizes = [len(intersection), len(only_pred), len(only_true)]
        labels = [f'命中\n{len(intersection)}', 
                 f'仅预测\n{len(only_pred)}', 
                 f'仅实际\n{len(only_true)}']
        colors_pie = ['green', 'orange', 'red']
        
        ax6.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 11})
        ax6.set_title(f'Top-{k} 命中情况分布', fontsize=13, fontweight='bold')
        
        # 7. 累积收益曲线
        ax7 = fig.add_subplot(gs[2, 1:])
        
        cumsum_pred = df_pred_sorted['true'].cumsum()
        cumsum_true = df_true_sorted['true'].cumsum()
        
        positions = range(1, len(cumsum_pred) + 1)
        
        ax7.plot(positions, cumsum_true, linewidth=2, 
                label='理想累积收益', color='green', marker='o', markersize=4)
        ax7.plot(positions, cumsum_pred, linewidth=2, 
                label='实际推荐累积收益', color='blue', marker='s', markersize=4)
        
        # 标记K值位置
        for k in k_values:
            ax7.axvline(x=k, color='red', linestyle='--', alpha=0.5)
            ax7.text(k, ax7.get_ylim()[1]*0.9, f'K={k}', 
                    rotation=0, fontsize=10, ha='center')
        
        ax7.set_xlabel('Top-K', fontsize=12)
        ax7.set_ylabel('累积收益率', fontsize=12)
        ax7.set_title('累积收益曲线 (理想 vs 实际)', fontsize=13, fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 填充差距区域
        ax7.fill_between(positions, cumsum_true, cumsum_pred, 
                        where=(cumsum_true >= cumsum_pred),
                        alpha=0.3, color='red', label='收益损失')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 排序分析图已保存: {save_path}")
        plt.close()
    
    def plot_comprehensive_dashboard(self, metrics_history, y_true, y_pred, 
                                    y_true_class, save_name="comprehensive_dashboard.png"):
        """
        绘制综合仪表盘 - 一图看全部指标
        
        参数:
            metrics_history: dict, 训练历史
            y_true: array, 实际收益率
            y_pred: array, 预测收益率
            y_true_class: array, 实际涨跌标签
        """
        print("\n[5] 绘制综合仪表盘...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        fig.suptitle('FinGAT 综合评价仪表盘', fontsize=18, fontweight='bold')
        
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        y_true_class = np.array(y_true_class).flatten()
        y_pred_class = (y_pred > 0).astype(int)
        
        # 第一行：训练曲线
        ax1 = fig.add_subplot(gs[0, :2])
        epochs = metrics_history.get('epochs', range(len(metrics_history.get('mae', []))))
        mae = metrics_history.get('mae', [])
        if mae:
            ax1.plot(epochs, mae, marker='o', linewidth=2, color=self.colors['MAE'])
            ax1.set_ylabel('MAE', fontsize=11)
            ax1.set_title('MAE训练曲线', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 2:])
        mrr = metrics_history.get('mrr', [])
        if mrr:
            ax2.plot(epochs, mrr, marker='^', linewidth=2, color=self.colors['MRR'])
            ax2.set_ylabel('MRR', fontsize=11)
            ax2.set_title('MRR训练曲线 ⭐', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # 第二行：预测散点图和混淆矩阵
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.scatter(y_true, y_pred, alpha=0.5, s=20, color=self.colors['MAE'])
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax3.set_xlabel('实际收益率', fontsize=11)
        ax3.set_ylabel('预测收益率', fontsize=11)
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        ax3.set_title(f'预测 vs 实际 (相关系数: {correlation:.3f})', 
                     fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(gs[1, 2:])
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true_class, y_pred_class)
        if HAS_SEABORN:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                       xticklabels=['跌', '涨'], yticklabels=['跌', '涨'])
        else:
            im = ax4.imshow(cm, cmap='Blues', aspect='auto')
            ax4.set_xticks([0, 1])
            ax4.set_yticks([0, 1])
            ax4.set_xticklabels(['跌', '涨'])
            ax4.set_yticklabels(['跌', '涨'])
            for i in range(2):
                for j in range(2):
                    ax4.text(j, i, str(cm[i, j]), ha='center', va='center',
                            color='white' if cm[i, j] > cm.max()/2 else 'black',
                            fontsize=12, fontweight='bold')
        accuracy = (y_true_class == y_pred_class).mean()
        ax4.set_title(f'混淆矩阵 (准确率: {accuracy:.3f})', 
                     fontsize=12, fontweight='bold')
        
        # 第三行：指标对比
        ax5 = fig.add_subplot(gs[2, 0])
        mae_val = np.mean(np.abs(y_true - y_pred))
        ax5.text(0.5, 0.5, f'{mae_val:.4f}', fontsize=36, 
                ha='center', va='center', color=self.colors['MAE'], fontweight='bold')
        ax5.text(0.5, 0.2, 'MAE', fontsize=16, ha='center', va='center')
        ax5.set_xlim([0, 1])
        ax5.set_ylim([0, 1])
        ax5.axis('off')
        
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.text(0.5, 0.5, f'{accuracy:.4f}', fontsize=36, 
                ha='center', va='center', color=self.colors['Accuracy'], fontweight='bold')
        ax6.text(0.5, 0.2, 'Accuracy', fontsize=16, ha='center', va='center')
        ax6.set_xlim([0, 1])
        ax6.set_ylim([0, 1])
        ax6.axis('off')
        
        # 计算MRR和Precision
        df = pd.DataFrame({'true': y_true, 'pred': y_pred})
        df_pred_sorted = df.sort_values('pred', ascending=False).reset_index(drop=True)
        df_true_sorted = df.sort_values('true', ascending=False).reset_index(drop=True)
        
        k = 5
        top_k_pred = set(df_pred_sorted.head(k).index)
        top_k_true = set(df_true_sorted.head(k).index)
        precision = len(top_k_pred & top_k_true) / k
        
        mrr_sum = 0
        for idx in df_true_sorted.head(k).index:
            pred_rank = df_pred_sorted.index[df_pred_sorted.index == idx][0] + 1
            mrr_sum += 1 / pred_rank
        mrr_val = mrr_sum
        
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.text(0.5, 0.5, f'{mrr_val:.4f}', fontsize=36, 
                ha='center', va='center', color=self.colors['MRR'], fontweight='bold')
        ax7.text(0.5, 0.2, 'MRR@5', fontsize=16, ha='center', va='center')
        ax7.set_xlim([0, 1])
        ax7.set_ylim([0, 1])
        ax7.axis('off')
        
        ax8 = fig.add_subplot(gs[2, 3])
        ax8.text(0.5, 0.5, f'{precision:.4f}', fontsize=36, 
                ha='center', va='center', color=self.colors['Precision'], fontweight='bold')
        ax8.text(0.5, 0.2, 'Precision@5', fontsize=16, ha='center', va='center')
        ax8.set_xlim([0, 1])
        ax8.set_ylim([0, 1])
        ax8.axis('off')
        
        # 第四行：分布和排名
        ax9 = fig.add_subplot(gs[3, :2])
        errors = y_pred - y_true
        ax9.hist(errors, bins=50, color=self.colors['MAE'], alpha=0.7, edgecolor='black')
        ax9.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax9.set_xlabel('预测误差', fontsize=11)
        ax9.set_ylabel('频数', fontsize=11)
        ax9.set_title('误差分布', fontsize=12, fontweight='bold')
        ax9.grid(True, alpha=0.3)
        
        ax10 = fig.add_subplot(gs[3, 2:])
        k_values = [5, 10, 20]
        precision_scores = []
        for k in k_values:
            top_k_pred = set(df_pred_sorted.head(k).index)
            top_k_true = set(df_true_sorted.head(k).index)
            precision_scores.append(len(top_k_pred & top_k_true) / k)
        
        ax10.bar(range(len(k_values)), precision_scores, 
                color=self.colors['Precision'], alpha=0.7)
        ax10.set_xticks(range(len(k_values)))
        ax10.set_xticklabels([f'K={k}' for k in k_values])
        ax10.set_ylabel('Precision', fontsize=11)
        ax10.set_title('不同K值的Precision', fontsize=12, fontweight='bold')
        ax10.set_ylim([0, 1])
        ax10.grid(True, alpha=0.3, axis='y')
        
        # 添加整体统计信息
        stats_text = f"""
        模型评估报告
        ━━━━━━━━━━━━━━━━━━━━━
        样本数: {len(y_true)}
        
        回归指标:
          MAE: {mae_val:.4f}
          相关系数: {correlation:.4f}
        
        分类指标:
          准确率: {accuracy:.4f}
        
        推荐指标:
          MRR@5: {mrr_val:.4f}
          Precision@5: {precision:.4f}
        """
        
        fig.text(0.98, 0.02, stats_text, fontsize=10, 
                verticalalignment='bottom', horizontalalignment='right',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 综合仪表盘已保存: {save_path}")
        plt.close()
    
    def generate_report(self, metrics_history, y_true, y_pred, y_true_class):
        """
        生成完整的可视化报告
        
        参数:
            metrics_history: dict, 训练历史
            y_true: array, 实际收益率
            y_pred: array, 预测收益率
            y_true_class: array, 实际涨跌标签
        """
        print("\n" + "="*60)
        print("开始生成完整可视化报告")
        print("="*60)
        
        # 1. 训练曲线
        self.plot_training_curves(metrics_history)
        
        # 2. 预测分析
        self.plot_prediction_analysis(y_true, y_pred)
        
        # 3. 分类分析
        self.plot_classification_analysis(y_true_class, y_pred)
        
        # 4. 排序分析
        self.plot_ranking_analysis(y_true, y_pred)
        
        # 5. 综合仪表盘
        self.plot_comprehensive_dashboard(metrics_history, y_true, y_pred, y_true_class)
        
        print("\n" + "="*60)
        print(f"✅ 所有可视化图表已生成完成！")
        print(f"📂 输出目录: {self.output_dir.absolute()}")
        print("="*60)
        
        # 列出生成的文件
        files = list(self.output_dir.glob("*.png"))
        print(f"\n生成的文件 ({len(files)}个):")
        for f in files:
            print(f"  - {f.name}")


def create_demo_data():
    """创建演示数据用于测试"""
    print("\n创建演示数据...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # 生成模拟的实际收益率
    y_true = np.random.randn(n_samples) * 0.05
    
    # 生成模拟的预测收益率（加入一些噪声）
    y_pred = y_true + np.random.randn(n_samples) * 0.02
    
    # 生成涨跌标签
    y_true_class = (y_true > 0).astype(int)
    
    # 生成训练历史
    n_epochs = 20
    metrics_history = {
        'epochs': list(range(1, n_epochs + 1)),
        'mae': [0.035 - i * 0.001 + np.random.rand() * 0.003 for i in range(n_epochs)],
        'accuracy': [0.55 + i * 0.005 + np.random.rand() * 0.02 for i in range(n_epochs)],
        'mrr': [0.60 + i * 0.01 + np.random.rand() * 0.03 for i in range(n_epochs)],
        'precision': [0.58 + i * 0.008 + np.random.rand() * 0.025 for i in range(n_epochs)]
    }
    
    return metrics_history, y_true, y_pred, y_true_class


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='FinGAT评价指标可视化系统')
    parser.add_argument('--mode', type=str, default='demo',
                       choices=['demo', 'realtime', 'analysis', 'compare', 'dashboard'],
                       help='运行模式')
    parser.add_argument('--data', type=str, default=None,
                       help='数据文件路径 (pickle格式)')
    parser.add_argument('--output', type=str, default='visualizations',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = MetricsVisualizer(output_dir=args.output)
    
    if args.mode == 'demo':
        print("\n=== 演示模式 ===")
        print("使用模拟数据生成示例可视化图表\n")
        
        # 创建演示数据
        metrics_history, y_true, y_pred, y_true_class = create_demo_data()
        
        # 生成完整报告
        visualizer.generate_report(metrics_history, y_true, y_pred, y_true_class)
        
    elif args.mode == 'analysis':
        if args.data is None:
            print("❌ 请提供数据文件路径: --data <path>")
            return
        
        print(f"\n=== 分析模式 ===")
        print(f"加载数据: {args.data}\n")
        
        # TODO: 加载实际数据并分析
        print("⚠️ 此模式需要实际的训练结果数据")
        print("提示: 请修改train.py保存预测结果和训练历史")
        
    else:
        print(f"⚠️ 模式 '{args.mode}' 暂未实现")
        print("当前支持的模式: demo")


if __name__ == "__main__":
    main()
