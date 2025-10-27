"""
数据准备脚本 - 为训练生成所需的边矩阵和pickle文件
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

print("=" * 60)
print("FinGAT 数据准备脚本")
print("=" * 60)

# 1. 读取类别信息
print("\n[步骤 1] 读取股票类别信息...")
category_file = "crypto_miners_category.csv"
stock_info = pd.read_csv(category_file)
print(f"✓ 读取 {len(stock_info)} 支股票")
print(f"✓ 类别数量: {stock_info['category'].nunique()}")
print("\n类别分布:")
print(stock_info['category'].value_counts())

# 2. 读取股票价格数据
print("\n[步骤 2] 读取股票价格数据...")
stock_data = {}
failed_stocks = []

for idx, row in stock_info.iterrows():
    company = row['company']
    category = row['category']
    try:
        df = pd.read_csv(f"./stock_data/{company}.csv")
        stock_data[company] = {
            'category': category,
            'stock_price': df
        }
        print(f"✓ {company} ({category}): {len(df)} 条记录")
    except Exception as e:
        print(f"✗ {company}: 读取失败 - {e}")
        failed_stocks.append(company)

if failed_stocks:
    print(f"\n警告: {len(failed_stocks)} 支股票读取失败: {failed_stocks}")
    stock_info = stock_info[~stock_info['company'].isin(failed_stocks)]

# 3. 对齐日期
print("\n[步骤 3] 对齐所有股票的交易日期...")
# 找到交集日期
all_dates = None
for company, data in stock_data.items():
    dates = set(data['stock_price']['Date'])
    if all_dates is None:
        all_dates = dates
    else:
        all_dates = all_dates.intersection(dates)

all_dates = sorted(list(all_dates))
print(f"✓ 共同交易日: {len(all_dates)} 天")
print(f"  起始日期: {all_dates[0]}")
print(f"  结束日期: {all_dates[-1]}")

# 过滤日期
for company in stock_data.keys():
    stock_data[company]["stock_price"] = stock_data[company]["stock_price"][
        stock_data[company]["stock_price"]['Date'].isin(all_dates)
    ].reset_index(drop=True)

# 4. 特征工程
print("\n[步骤 4] 特征工程...")

# 归一化收盘价
print("  - 价格归一化")
for company in stock_data.keys():
    scaler = StandardScaler()
    close_price = np.array(stock_data[company]["stock_price"]["Close"]).reshape(-1, 1)
    normalized = scaler.fit_transform(close_price).ravel()
    stock_data[company]["stock_price"]["nor_close"] = normalized

# 计算收益率
print("  - 计算收益率")
for company in stock_data.keys():
    prices = np.array(stock_data[company]["stock_price"]["Close"])
    returns = [0] + [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
    stock_data[company]["stock_price"]["return ratio"] = returns

# 相对价格特征
print("  - 相对价格特征 (c_open, c_high, c_low)")
for company in stock_data.keys():
    df = stock_data[company]["stock_price"]
    df["c_open"] = (df["Open"] / df["Close"]) - 1
    df["c_high"] = (df["High"] / df["Close"]) - 1
    df["c_low"] = (df["Low"] / df["Close"]) - 1

# 移动平均线
print("  - 移动平均线 (5/10/15/20/25/30天)")
for company in stock_data.keys():
    close = stock_data[company]["stock_price"]["Close"]
    for window in [5, 10, 15, 20, 25, 30]:
        ma = []
        for day in range(len(close)):
            if day >= window - 1:
                ma.append((close.iloc[day-window+1:day+1].mean() / close.iloc[day]) - 1)
            else:
                ma.append(0)
        stock_data[company]["stock_price"][f"{window}-days"] = ma

# 类别编码
print("  - 类别One-hot编码")
categories = stock_info["category"].unique()
for company in stock_data.keys():
    company_category = stock_data[company]['category']
    for cat in categories:
        stock_data[company]["stock_price"][f"label_{cat}"] = int(cat == company_category)

# 5. 提取特征
print("\n[步骤 5] 提取特征矩阵...")
features = {}
for company in stock_data.keys():
    # 从第30天开始（移动平均需要）
    features[company] = stock_data[company]["stock_price"].iloc[30:, 7:].reset_index(drop=True)

print(f"✓ 特征维度: {features[list(features.keys())[0]].shape[1]}")
print(f"✓ 时间长度: {len(features[list(features.keys())[0]])} 天")

# 6. 生成标签
print("\n[步骤 6] 生成涨跌标签...")
Y_buy_or_not = {}
for company in stock_data.keys():
    Y_buy_or_not[company] = (features[company]['return ratio'] >= 0) * 1

# 7. 划分训练测试集
print("\n[步骤 7] 划分训练/测试集...")
train_ratio = 0.2
days = len(features[list(features.keys())[0]])
train_days = int(days * train_ratio)
test_days = days - train_days

print(f"  训练集: {train_days} 天")
print(f"  测试集: {test_days} 天")

train_data = {}
test_data = {}
train_Y = {}
test_Y = {}

for company in stock_data.keys():
    train_data[company] = features[company].iloc[:train_days, :]
    train_Y[company] = Y_buy_or_not[company][:train_days]
    test_data[company] = features[company].iloc[train_days:, :]
    test_Y[company] = Y_buy_or_not[company][train_days:]

# 8. 生成滑动窗口
print("\n[步骤 8] 生成滑动窗口...")

def create_sliding_windows(data_dict, y_dict, num_weeks=4):
    """创建滑动窗口数据"""
    companies = list(data_dict.keys())
    data_length = len(data_dict[companies[0]])
    
    result = {}
    
    # 为每周创建数据
    for w in range(num_weeks):
        weekly_data = []
        for time_idx in range(data_length - 7 - (num_weeks - 2) - 1):
            weekly_stocks = []
            for company in companies:
                df = data_dict[company]
                stock_week = df.iloc[time_idx + w : time_idx + w + 7, :].values
                weekly_stocks.append(stock_week)
            weekly_data.append(weekly_stocks)
        result[f'x{w+1}'] = np.array(weekly_data)
        print(f"  x{w+1} 形状: {result[f'x{w+1}'].shape}")
    
    # 生成标签
    y_returns = []
    y_classes = []
    for time_idx in range(data_length - 7 - (num_weeks - 2) - 1):
        returns = []
        classes = []
        for company in companies:
            returns.append(data_dict[company]["return ratio"].iloc[time_idx + num_weeks - 1 + 7])
            classes.append(y_dict[company].iloc[time_idx + num_weeks - 1 + 7])
        y_returns.append(returns)
        y_classes.append(classes)
    
    result['y_return ratio'] = np.array(y_returns)
    result['y_up_or_down'] = np.array(y_classes)
    print(f"  y_return ratio 形状: {result['y_return ratio'].shape}")
    print(f"  y_up_or_down 形状: {result['y_up_or_down'].shape}")
    
    return result

train_windows = create_sliding_windows(train_data, train_Y, num_weeks=4)
test_windows = create_sliding_windows(test_data, test_Y, num_weeks=4)

# 9. 保存数据
print("\n[步骤 9] 保存数据到pickle文件...")
data = {
    "train": train_windows,
    "test": test_windows
}

output_file = "Taiwan_model_data_10_best.pickle"
with open(output_file, "wb") as f:
    pickle.dump(data, f)
print(f"✓ 数据已保存到 {output_file}")

# 10. 生成图边矩阵
print("\n[步骤 10] 生成图边矩阵...")

# 建立公司到索引的映射
company_list = list(stock_data.keys())
company_to_idx = {company: idx for idx, company in enumerate(company_list)}
category_to_companies = {}
for company, data in stock_data.items():
    cat = data['category']
    if cat not in category_to_companies:
        category_to_companies[cat] = []
    category_to_companies[cat].append(company)

print(f"  公司总数: {len(company_list)}")
print(f"  类别数: {len(category_to_companies)}")

# Inner edge: 同类别内的全连接
print("  - 生成inner edge (类别内连接)")
inner_edges = []
for category, companies in category_to_companies.items():
    for i, comp1 in enumerate(companies):
        for j, comp2 in enumerate(companies):
            if i != j:
                idx1 = company_to_idx[comp1]
                idx2 = company_to_idx[comp2]
                inner_edges.append([idx1, idx2])

inner_edge = np.array(inner_edges)
np.save("Taiwan_inner_edge.npy", inner_edge)
print(f"    ✓ 保存 Taiwan_inner_edge.npy: {inner_edge.shape}")

# Outer edge: 类别间的连接（所有类别相互连接）
print("  - 生成outer edge (类别间连接)")
category_list = list(category_to_companies.keys())
category_representatives = {}

# 为每个类别选一个代表节点（第一个公司）
for cat in category_list:
    category_representatives[cat] = company_to_idx[category_to_companies[cat][0]]

outer_edges = []
for i, cat1 in enumerate(category_list):
    for j, cat2 in enumerate(category_list):
        if i != j:
            outer_edges.append([category_representatives[cat1], category_representatives[cat2]])

outer_edge = np.array(outer_edges)
np.save("Taiwan_outer_edge.npy", outer_edge)
print(f"    ✓ 保存 Taiwan_outer_edge.npy: {outer_edge.shape}")

# Edge 10: K近邻图（基于类别相似度的10邻居）
print("  - 生成edge_10 (10-邻居图)")
edge_10 = []
for i in range(len(company_list)):
    # 为每个节点创建10个连接（简化版本）
    neighbors = min(10, len(company_list) - 1)
    for j in range(neighbors):
        target = (i + j + 1) % len(company_list)
        edge_10.append([i, target])

edge_10 = np.array(edge_10)
np.save("edge_10.npy", edge_10)
print(f"    ✓ 保存 edge_10.npy: {edge_10.shape}")

# Inner 20 edge: 类别内的小型边集（用于池化）
print("  - 生成Taiwan_inner_edge20 (池化用边)")
inner20_edges = []
for category, companies in category_to_companies.items():
    # 为每个类别创建一个小图
    local_nodes = len(companies)
    for i in range(local_nodes):
        for j in range(local_nodes):
            if i != j:
                inner20_edges.append([i, j])

# 如果需要固定为20个节点
if len(inner20_edges) == 0:
    # 创建一个默认的20节点全连接图
    for i in range(20):
        for j in range(20):
            if i != j:
                inner20_edges.append([i, j])

inner20_edge = np.array(inner20_edges)
np.save("Taiwan_inner_edge20.npy", inner20_edge)
print(f"    ✓ 保存 Taiwan_inner_edge20.npy: {inner20_edge.shape}")

# 11. 总结
print("\n" + "=" * 60)
print("数据准备完成！")
print("=" * 60)
print("\n生成的文件:")
print("  1. Taiwan_model_data_10_best.pickle - 训练和测试数据")
print("  2. Taiwan_inner_edge.npy - 类别内边矩阵")
print("  3. Taiwan_outer_edge.npy - 类别间边矩阵")
print("  4. edge_10.npy - 10邻居边矩阵")
print("  5. Taiwan_inner_edge20.npy - 池化用边矩阵")
print("\n下一步: 运行训练脚本")
print("  python train.py --model CAT --device cpu --epochs 10")
print("\n" + "=" * 60)
