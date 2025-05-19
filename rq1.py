import pandas as pd
import time, os
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

time1=time.time()
folder_path = '/home/ylqiu/datamining_filter/'
parquet_files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]

# 初始化空的DataFrame用于存储数据
df = pd.DataFrame()
for file in parquet_files:
    file_path = os.path.join(folder_path, file)
    data = pd.read_parquet(file_path, engine='pyarrow')
    df=pd.concat([df, data])

transactions = df['item_list'].tolist()

# 使用TransactionEncoder进行one-hot编码
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# 找出支持度≥0.02的频繁项集
frequent_itemsets = apriori(df_encoded, min_support=0.02, use_colnames=True)
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)
print("频繁项集:")
print(frequent_itemsets)
print(time.time()-time1)

# 生成关联规则
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
rules = rules.sort_values(by='confidence', ascending=False)
print("\n关联规则(置信度≥0.5):")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# 将频繁项集保存到CSV
frequent_itemsets.to_csv('frequent_itemsets.csv', index=False)
rules.to_csv('association_rules.csv', index=False)