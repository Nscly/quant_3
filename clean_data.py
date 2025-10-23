import pandas as pd
import numpy as np
import math
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder


# the information of stocks (i.e. name & category of sector).
category_df = pd.read_csv("crypto_miners_category.csv")

# original information of stock price.
stock_data = {}
for company in category_df["company"]:
    da = {}
    da['category'] = category_df[category_df.company==company]['category'].iloc[0]
    da['stock_price'] = pd.read_csv("./stock_data/%s.csv" % (company))
    stock_data[company] = da

# Let all stocks have the same dates - use the stock with minimum dates
min_dates = None
min_len = float('inf')
for company in stock_data.keys():
    dates = stock_data[company]["stock_price"]['Date']
    if len(dates) < min_len:
        min_len = len(dates)
        min_dates = dates

need_day = np.array(min_dates)
for company in stock_data.keys():
    stock_data[company]["stock_price"] = stock_data[company]["stock_price"][stock_data[company]["stock_price"]['Date'].isin(need_day)].reset_index(drop=True)
    stock_data[company]["stock_price"].index = stock_data[company]["stock_price"]['Date']


### feature engineering ###

# normalize stock price
normalize_scalar = {}
for company in stock_data.keys():
    scaler = StandardScaler()
    nor_data = scaler.fit_transform(np.array(stock_data[company]["stock_price"]["Close"]).reshape(-1,1)).ravel()
    stock_data[company]["stock_price"]["nor_close"] = nor_data
    normalize_scalar[company] = scaler

# calculate return ratio
for company in stock_data.keys():
    return_ratio = []
    data = np.array(stock_data[company]["stock_price"]["Close"])
    for i in range(len(data)):
        if i == 0:
            return_ratio.append(0)
        else:
            return_ratio.append((data[i]-data[i-1])/data[i-1])
    stock_data[company]["stock_price"]["return ratio"] = return_ratio


# feature of c_open / c_high / c_low
for company in stock_data.keys():
    function = lambda x,y: (x/y)-1
    data = stock_data[company]["stock_price"]
    data["c_open"] = list(map(function, data["Open"], data["Close"]))
    data["c_high"] = list(map(function, data["High"], data["Close"]))
    data["c_low"] = list(map(function, data["Low"], data["Close"]))


# 5 / 10 / 15 / 20 / 25 / 30 days moving average
for company in stock_data.keys():
    data = stock_data[company]["stock_price"]["Close"]
    for i in [5,10,15,20,25,30]:
        q = []
        for day in range(len(data)):
            if day >= i-1:
                q.append((np.mean(data.iloc[day-i+1:day+1])/data.iloc[day])-1)
            else:
                q.append(0)
        stock_data[company]["stock_price"]["%s-days" % (i)] = q


# category of sector (one hot encoding)
unique_categories = category_df["category"].unique()
for company in stock_data.keys():
    for cat in unique_categories:
        cate = stock_data[company]['category']
        if cat != cate:
            stock_data[company]["stock_price"]["label_%s" % (cat)] = 0
        else:
            stock_data[company]["stock_price"]["label_%s" % (cat)] = 1


# total feature - start from row 30 to skip initial moving average zeros
features = {}
for company in stock_data.keys():
    features[company] = stock_data[company]["stock_price"].iloc[30:, 7:].reset_index(drop=True)

# movement of stock (1 if positive return, 0 if negative)
Y_buy_or_not = {}
for company in stock_data.keys():
    Y_buy_or_not[company] = (features[company]['return ratio']>=0)*1


## Training & Testing split ##
train_size = 0.8  # Fixed: was 0.2, should be 0.8 for training
test_size = 0.2   # Fixed: was 0.8, should be 0.2 for testing
days = len(features[list(stock_data.keys())[0]])

train_day = int(days*train_size)

# data of training set and testing set
train_data = {}
test_data = {}
train_Y_buy_or_not = {}
test_Y_buy_or_not = {}

for company in stock_data.keys():
    train_data[company] = features[company].iloc[:train_day, :]
    train_Y_buy_or_not[company] = Y_buy_or_not[company][:train_day]
    test_data[company] = features[company].iloc[train_day:, :]
    test_Y_buy_or_not[company] = Y_buy_or_not[company][train_day:]


# week represents the number of input weeks
def before_day(week):
    # train
    train = {}
    companies_list = list(stock_data.keys())
    
    for w in range(week):
        train_x = []
        for tr_ind in range(len(train_data[companies_list[0]])-7-(week-2)-1):
            tr = []
            for company in companies_list:
                data = train_data[company]
                tr.append(data.iloc[tr_ind+w:tr_ind+w+7, :].values)
            train_x.append(tr)
        train["x%s" % (w+1)] = np.array(train_x)
        
    train_y1, train_y2 = [], []
    for tr_ind in range(len(train_data[companies_list[0]])-7-(week-2)-1):
        tr_y1, tr_y2 = [], []
        for company in companies_list:
            data = train_data[company]
            tr_y1.append(data["return ratio"].iloc[tr_ind+(week-1)+7])
            tr_y2.append(train_Y_buy_or_not[company].iloc[tr_ind+(week-1)+7])
        train_y1.append(tr_y1)
        train_y2.append(tr_y2)
    train['y_return ratio'] = np.array(train_y1)
    train["y_up_or_down"] = np.array(train_y2)

    # test
    test = {}
    for w in range(week):
        test_x = []
        for te_ind in range(len(test_data[companies_list[0]])-7-(week-2)-1):
            te = []
            for company in companies_list:
                data = test_data[company]
                te.append(data.iloc[te_ind+w:te_ind+w+7, :].values)
            test_x.append(te)
        test['x%s' % (w+1)] = np.array(test_x)
    
    test_y1, test_y2 = [], []
    for te_ind in range(len(test_data[companies_list[0]])-7-(week-2)-1):
        te_y1, te_y2 = [], []
        for company in companies_list:
            data = test_data[company]
            te_y1.append(data["return ratio"].iloc[te_ind+(week-1)+7])
            te_y2.append(test_Y_buy_or_not[company].iloc[te_ind+(week-1)+7])
        test_y1.append(te_y1)
        test_y2.append(te_y2)
    test['y_return ratio'] = np.array(test_y1)
    test["y_up_or_down"] = np.array(test_y2)
    
    data = {"train": train, "test": test}
    
    return data


def build_graph_edges(category_df, stock_data):
    """
    Build graph edge matrices for the stock network:
    - inner_edge: edges within the same category (intra-category connections)
    - outer_edge: edges between different categories (inter-category connections)
    """
    companies_list = list(stock_data.keys())
    num_stocks = len(companies_list)
    
    # Create company to index mapping
    company_to_idx = {comp: idx for idx, comp in enumerate(companies_list)}
    
    # Create category to companies mapping
    category_to_companies = {}
    for company in companies_list:
        cat = stock_data[company]['category']
        if cat not in category_to_companies:
            category_to_companies[cat] = []
        category_to_companies[cat].append(company)
    
    # Build inner edges (within same category) - fully connected within category
    inner_edges = []
    for cat, companies in category_to_companies.items():
        for i, comp1 in enumerate(companies):
            for comp2 in companies:
                if comp1 != comp2:  # No self-loops
                    idx1 = company_to_idx[comp1]
                    idx2 = company_to_idx[comp2]
                    inner_edges.append([idx1, idx2])
    
    inner_edge = np.array(inner_edges) if inner_edges else np.array([]).reshape(0, 2)
    
    # Build category-level graph for outer edges
    # First, create category nodes by averaging stock features
    categories = list(category_to_companies.keys())
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    
    # Outer edges: fully connected category graph
    outer_edges = []
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            if i != j:  # No self-loops
                outer_edges.append([i, j])
    
    outer_edge = np.array(outer_edges) if outer_edges else np.array([]).reshape(0, 2)
    
    # Build correlation-based edges for inner10 and inner20
    # Calculate correlation matrix between stocks using historical returns
    returns_matrix = []
    for company in companies_list:
        returns = features[company]['return ratio'].values
        returns_matrix.append(returns)
    
    returns_matrix = np.array(returns_matrix)
    correlation_matrix = np.corrcoef(returns_matrix)
    
    # Inner edge with top-10 most correlated stocks
    inner10_edges = []
    for i in range(num_stocks):
        correlations = correlation_matrix[i]
        # Get indices of top 10 correlated stocks (excluding itself)
        top_k_indices = np.argsort(correlations)[::-1][1:11]  # Exclude self, get top 10
        for j in top_k_indices:
            inner10_edges.append([i, j])
    
    inner10_edge = np.array(inner10_edges)
    
    # Inner edge with top-20 most correlated stocks (for pooling)
    inner20_edges = []
    for i in range(num_stocks):
        correlations = correlation_matrix[i]
        # Get indices of top 20 correlated stocks (excluding itself)
        top_k_indices = np.argsort(correlations)[::-1][1:21]  # Exclude self, get top 20
        for j in top_k_indices:
            inner20_edges.append([i, j])
    
    inner20_edge = np.array(inner20_edges)
    
    return inner_edge, inner10_edge, inner20_edge, outer_edge, category_to_companies


if __name__ == "__main__":
    print("=" * 50)
    print("Starting data preprocessing...")
    print("=" * 50)
    
    # Generate data with 4 weeks
    week_num = 4
    print(f"\nGenerating sliding window data with {week_num} weeks...")
    data = before_day(week_num)
    
    # Build graph edges
    print("\nBuilding graph structures...")
    inner_edge, inner10_edge, inner20_edge, outer_edge, category_map = build_graph_edges(category_df, stock_data)
    
    print(f"\nGraph Statistics:")
    print(f"  - Total stocks: {len(stock_data)}")
    print(f"  - Number of categories: {len(category_map)}")
    print(f"  - Inner edges (same category): {len(inner_edge)}")
    print(f"  - Inner10 edges (top-10 correlated): {len(inner10_edge)}")
    print(f"  - Inner20 edges (top-20 correlated): {len(inner20_edge)}")
    print(f"  - Outer edges (between categories): {len(outer_edge)}")
    
    print("\nCategory distribution:")
    for cat, companies in category_map.items():
        print(f"  - {cat}: {len(companies)} stocks")
    
    # Save processed data
    print("\nSaving processed data...")
    with open("stock_data_processed.pickle", "wb") as f:
        pickle.dump(data, f)
    
    # Save graph edge matrices
    np.save("inner_edge.npy", inner_edge)
    np.save("inner10_edge.npy", inner10_edge)
    np.save("inner20_edge.npy", inner20_edge)
    np.save("outer_edge.npy", outer_edge)
    
    # Save category mapping
    with open("category_mapping.pickle", "wb") as f:
        pickle.dump({
            'category_to_companies': category_map,
            'num_stocks': len(stock_data),
            'num_categories': len(category_map)
        }, f)
    
    print("\n" + "=" * 50)
    print("Data preprocessing completed!")
    print("=" * 50)
    print("\nGenerated files:")
    print("  - stock_data_processed.pickle")
    print("  - inner_edge.npy")
    print("  - inner10_edge.npy") 
    print("  - inner20_edge.npy")
    print("  - outer_edge.npy")
    print("  - category_mapping.pickle")
    print("\nTraining data shape:")
    print(f"  - x1: {data['train']['x1'].shape}")
    print(f"  - y_return ratio: {data['train']['y_return ratio'].shape}")
    print(f"  - y_up_or_down: {data['train']['y_up_or_down'].shape}")
    print("\nTest data shape:")
    print(f"  - x1: {data['test']['x1'].shape}")
    print(f"  - y_return ratio: {data['test']['y_return ratio'].shape}")
    print(f"  - y_up_or_down: {data['test']['y_up_or_down'].shape}")
    print("\nReady for training!")
