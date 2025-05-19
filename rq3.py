import pandas as pd
import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

time1=time.time()
data = pd.DataFrame()
file_path = "/home/ylqiu/datamining/part-00001simpleRQ1.parquet"
df = pd.read_parquet(file_path, engine='pyarrow').loc[1:1000]
def get_season(date):
    month = date.month
    if 3 <= month <= 5:
        return 'Spring'
    elif 6 <= month <= 8:
        return 'Summer'
    elif 9 <= month <= 11:
        return 'Autumn'
    else:
        return 'Winter'

df['purchase_date'] = pd.to_datetime(df['purchase_date'])
df['purchase_season'] = df['purchase_date'].apply(get_season)


def create_transaction(row):
    return [str(row['item_list']), "季节_"+row['purchase_season']]
transactions = df.apply(create_transaction, axis=1).tolist()

# 使用TransactionEncoder进行编码
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# 找出支持度≥0.01的频繁项集
frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)
print(frequent_itemsets)


# 生成关联规则(置信度≥0.6)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
payment_rules = rules[
    rules['antecedents'].apply(lambda x: any(item.startswith('季节_') for item in x)) |
    rules['consequents'].apply(lambda x: any(item.startswith('季节_') for item in x))
]
payment_rules = payment_rules.sort_values(by='confidence', ascending=False)
print("订单状态与商品的关联规则(支持度≥0.01，置信度≥0.6):")
print(payment_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])