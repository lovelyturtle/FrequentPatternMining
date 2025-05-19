import pandas as pd
import time, os
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

time1=time.time()
folder_path = '/home/ylqiu/datamining_filter/'
parquet_files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]
df = pd.DataFrame()
for file in parquet_files:
    file_path = os.path.join(folder_path, file)
    data = pd.read_parquet(file_path, engine='pyarrow')
    df=pd.concat([df, data])

def create_transaction(row):
    return [str(row['item_list']), "订单状态_"+row['purchase_status']]
transactions = df.apply(create_transaction, axis=1).tolist()

# 使用TransactionEncoder进行编码
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# 找出支持度≥0.005的频繁项集
frequent_itemsets = apriori(df_encoded, min_support=0.005, use_colnames=True)
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)
print(frequent_itemsets)


# 生成关联规则(置信度≥0.3)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
payment_rules = rules[
    rules['antecedents'].apply(lambda x: any(item.startswith('订单状态_') for item in x)) |
    rules['consequents'].apply(lambda x: any(item.startswith('订单状态_') for item in x))
]
payment_rules = payment_rules.sort_values(by='confidence', ascending=False)
print("订单状态与商品的关联规则(支持度≥0.005，置信度≥0.3):")
print(payment_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])