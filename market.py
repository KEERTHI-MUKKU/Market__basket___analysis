import pandas as pd
df = pd.read_csv('Groceries_dataset.csv')
df.head()
df[‘single_transaction’] = df[‘Member_number’].astype(str)+’_’+df[‘Date’].astype(str)

df.head()
df2 = pd.crosstab(df['single_transaction'], df['itemDescription'])
df2.head()
def encode(item_freq):
    res = 0
    if item_freq > 0:
        res = 1
    return res
    
basket_input = df2.applymap(encode)
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

frequent_itemsets = apriori(basket_input, min_support=0.001, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift")

rules.head()
rules.sort_values(["support", "confidence","lift"],axis = 0, ascending = False).head(8)
