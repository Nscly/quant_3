# FinGAT: A Financial Graph Attention Network to Recommend Top-K Profitable Stocks

This is an implementation of the FinGAT model for stock recommendation based on graph neural networks.

Paper: [FinGAT: A Financial Graph Attention Network to Recommend Top-K Profitable Stocks](https://arxiv.org/abs/2106.10159)

## Overview

FinGAT uses Graph Attention Networks (GAT) to model relationships between stocks and their sectors. The model:
- Encodes time-series stock data using GRU with attention
- Builds graph connections between stocks in the same category (intra-category edges)
- Builds graph connections between different categories (inter-category edges)
- Uses correlation-based edges to capture stock co-movements
- Predicts stock returns, direction (up/down), and provides top-K recommendations

## Requirements

```bash
pip install pandas numpy scikit-learn torch torch-geometric
```

Tested with:
- Python 3.8+
- PyTorch 1.0+
- pandas 2.0+
- numpy 1.16+
- scikit-learn 1.0+
- torch-geometric 2.0+

## Data Format

Your data should include:
1. **Category CSV file**: `crypto_miners_category.csv` with columns:
   - `company`: Stock ticker symbol
   - `category`: Sector/category name

2. **Stock CSV files**: Individual CSV files in `stock_data/` directory with columns:
   - `Date`: Trading date
   - `Open`, `High`, `Low`, `Close`: Price data
   - `Volume`: Trading volume

## Usage

### Step 1: Data Preprocessing

Run the data preprocessing script to:
- Load and align stock data
- Engineer features (moving averages, return ratios, etc.)
- Build graph structures (inner/outer edges)
- Create train/test splits
- Save processed data

```bash
python clean_data.py
```

This will generate:
- `stock_data_processed.pickle`: Processed training/test data
- `inner_edge.npy`: Intra-category graph edges (fully connected within categories)
- `inner10_edge.npy`: Top-10 correlation-based edges
- `inner20_edge.npy`: Top-20 correlation-based edges
- `outer_edge.npy`: Inter-category graph edges (fully connected between categories)
- `category_mapping.pickle`: Category and stock metadata

### Step 2: Train the Model

Run training with default parameters:

```bash
python train.py
```

Or customize hyperparameters:

```bash
python train.py --model CAT --epochs 20 --lr 0.01 --dim 32 --device cpu
```

Available arguments:
- `--data`: Path to processed data pickle file (default: `stock_data_processed.pickle`)
- `--model`: Model architecture - choose from:
  - `CG`: Categorical Graph (basic)
  - `CAT`: Categorical Graph with Attention (recommended)
  - `CPool`: Categorical Graph with Pooling
- `--epochs`: Number of training epochs (default: 20)
- `--dim`: Hidden dimension size (default: 16)
- `--lr`: Learning rate (default: 0.05)
- `--l2`: L2 regularization weight (default: 0)
- `--alpha`: Weight for MAE loss (default: 1)
- `--beta`: Weight for classification loss (default: 1)
- `--gamma`: Weight for ranking loss (default: 1)
- `--device`: Device for training - `cpu` or `cuda:0` (default: cpu)
- `--use_gru`: Whether to use GRU for weekly encoding (default: False)
- `--week_num`: Number of weeks to aggregate (default: 3)

### Example Commands

Train with CAT model on CPU:
```bash
python train.py --model CAT --epochs 10 --device cpu
```

Train with CG model with custom loss weights:
```bash
python train.py --model CG --alpha 2 --beta 1 --gamma 0.5 --epochs 15
```

Train with larger hidden dimensions:
```bash
python train.py --model CAT --dim 64 --epochs 20
```

## Model Architecture

### Feature Engineering
- Normalized closing prices
- Return ratios (day-to-day changes)
- Relative price features (open/close, high/close, low/close)
- Moving averages (5, 10, 15, 20, 25, 30 days)
- One-hot encoded sector categories

### Graph Construction
1. **Inner Graph (Intra-category)**: Connects stocks within the same sector
2. **Outer Graph (Inter-category)**: Connects different sectors
3. **Correlation Graph**: Connects highly correlated stocks based on historical returns

### Network Components
- **Sequence Encoder**: GRU + Attention for time-series features
- **Graph Attention**: GAT layers for stock relationships
- **Multi-task Learning**: Joint optimization of:
  - Regression: Predict return ratio (MAE loss)
  - Classification: Predict up/down movement (BCE loss)
  - Ranking: Learn relative ordering (pairwise ranking loss)

## Evaluation Metrics

The model is evaluated on:
- **MAE**: Mean Absolute Error for return prediction
- **Accuracy**: Classification accuracy for up/down movement
- **MRR**: Mean Reciprocal Rank for top-K recommendations
- **Precision@K**: Precision at K for recommended stocks (K=5, 10, 20)

## Results

Results are printed for each epoch showing:
```
Epoch X - REG Loss: X.XXXX CLS Loss: X.XXXX RANK Loss: X.XXXX Total Loss: X.XXXX
Results for k=[5, 10, 20]: [[mae, acc, mrr, precision], ...]
```

Best results are saved based on MRR metric.

## Project Structure

```
.
├── clean_data.py           # Data preprocessing and graph construction
├── train.py                # Model training and evaluation
├── parse_arg.py            # Command-line argument parsing
├── model/
│   └── graph_pool.py       # Model architectures (CG, CAT, CPool)
├── crypto_miners_category.csv  # Stock category mapping
├── stock_data/             # Stock price CSV files
└── README.md               # This file
```

## Notes

- The default configuration uses CPU. For GPU training, ensure CUDA is available and use `--device cuda:0`
- Training time depends on the number of stocks, categories, and epochs
- The model automatically handles different numbers of stocks and categories
- Graph edges are built based on category membership and historical correlations
- Data is split 80% training / 20% testing by default

## Citation

If you use this code, please cite:

```
@article{fingat2021,
  title={FinGAT: A Financial Graph Attention Network to Recommend Top-K Profitable Stocks},
  author={},
  journal={arXiv preprint arXiv:2106.10159},
  year={2021}
}
```
