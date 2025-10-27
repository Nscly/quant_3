# FinGAT: A Financial Graph Attention Network to Recommend Top-K Profitable Stocks

This is our implementation for the [paper](https://arxiv.org/abs/2106.10159):
FinGAT: A Financial Graph Attention Network to Recommend Top-K Profitable Stocks

---

## 📚 Documentation (中文文档)

**🎯 Quick Start / 快速开始:**
- **[SUMMARY_CN.md](./SUMMARY_CN.md)** - 完整分析总结报告
- **[README_CN.md](./README_CN.md)** - 中文快速入门指南
- **[DOCS_INDEX.md](./DOCS_INDEX.md)** - 文档导航索引

**📖 Detailed Guides / 详细指南:**
- **[TRAINING_GUIDE.md](./TRAINING_GUIDE.md)** - 训练流程详解
- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - 技术架构文档
- **[TRAINING_ANALYSIS.md](./TRAINING_ANALYSIS.md)** - 代码分析报告

**⚙️ Configuration / 配置:**
- **[train_configs.json](./train_configs.json)** - 训练配置示例

---

## 🚀 Quick Start

### Method 1: Using Script (Recommended)
```bash
# Activate virtual environment
source .venv/bin/activate

# One-command training
./quick_train.sh
```

### Method 2: Manual Steps
```bash
# Activate virtual environment
source .venv/bin/activate

# Step 1: Prepare data
python prepare_data.py

# Step 2: Train model
python train.py --model CAT --epochs 20 --device cpu
```

---

## 📊 Current Dataset

- **Stocks**: 27 stocks across 8 categories
- **Categories**: Bitcoin Mining, Semiconductors, Energy, Financials, Cloud/Big Tech, Data Infrastructure
- **Time Range**: 108 trading days
- **Features**: 19 dimensions (price, returns, technical indicators, category encoding)

---

## 🧠 Model Architectures

### 1. CG (Categorical Graph) - Baseline
- Fast training
- Basic graph attention
- Good for quick validation

### 2. CAT (Categorical Graph Attention) ⭐ Recommended
- Dual-layer graph structure (Inner + Outer GAT)
- Best performance
- Recommended for production

### 3. CPool (Categorical Graph Pool) - Advanced
- Includes graph pooling (SAGPooling)
- Higher computational cost
- Good for research

---

## Requirements
* pytorch >= 1.0.0
* torch-geometric
* numpy >= 1.16.4
* pandas >= 0.25.3
* scikit-learn
* matplotlib

---

## Model Architecture
![](https://i.imgur.com/lkCA1Rt.png)

---

## How to Train the Model

### Quick Test (5 minutes)
```bash
python train.py --model CG --epochs 5 --device cpu
```

### Standard Training (30 minutes on CPU / 5 minutes on GPU)
```bash
python train.py --model CAT --epochs 20 --dim 32 --lr 0.01 --device cuda:0
```

### Full Training (2-3 hours)
```bash
python train.py --model CAT --epochs 100 --dim 64 --lr 0.005 \
  --l2 1e-5 --use_gru True --week_num 4 --device cuda:0
```

### Key Parameters
```
--model:    Model type (CG/CAT/CPool)
--epochs:   Number of training epochs (default: 20)
--dim:      Hidden dimension (default: 16)
--lr:       Learning rate (default: 0.05)
--l2:       L2 regularization (default: 0)
--alpha:    Weight for MAE loss (default: 1.0)
--beta:     Weight for classification loss (default: 1.0)
--gamma:    Weight for ranking loss (default: 1.0)
--device:   Training device (cpu/cuda:0)
--use_gru:  Use GRU for weekly encoding (default: False)
--week_num: Number of input weeks (default: 3)
```

---

## 📈 Evaluation Metrics

- **MAE**: Mean Absolute Error (regression accuracy)
- **Accuracy**: Classification accuracy (up/down prediction)
- **MRR**: Mean Reciprocal Rank (ranking quality) ⭐ Primary metric
- **Precision@K**: Top-K recommendation precision (K=5,10,20)

---

## 📝 Files Structure

```
FinGAT/
├── README.md                    # This file
├── README_CN.md                 # Chinese guide
├── DOCS_INDEX.md               # Documentation index
├── TRAINING_GUIDE.md           # Training guide
├── ARCHITECTURE.md             # Architecture docs
├── TRAINING_ANALYSIS.md        # Analysis report
├── SUMMARY_CN.md               # Summary report
├── train_configs.json          # Config examples
│
├── stock_data/                 # Stock CSV files
├── crypto_miners_category.csv  # Category labels
│
├── prepare_data.py            # Data preparation script
├── clean_data.py              # Original data cleaning
├── train.py                   # Training script
├── parse_arg.py              # Argument parser
├── quick_train.sh            # Quick training script
│
└── model/
    └── graph_pool.py         # Model definitions
```

---

## Result
![](https://i.imgur.com/ANEXmfH.png)
![](https://i.imgur.com/e8KmLKU.png)
![](https://i.imgur.com/DGuClLM.png)

---

## 🎓 Citation

If you use this code, please cite the original paper:
```
@article{fingat2021,
  title={FinGAT: Financial Graph Attention Networks for Recommending Top-K Profitable Stocks},
  author={...},
  journal={arXiv preprint arXiv:2106.10159},
  year={2021}
}
```

---

## 📞 Getting Help

For detailed information, please refer to:
- **[DOCS_INDEX.md](./DOCS_INDEX.md)** - Documentation navigation
- **[README_CN.md](./README_CN.md)** - Chinese quick start guide
- **[TRAINING_ANALYSIS.md](./TRAINING_ANALYSIS.md)** - Comprehensive analysis


