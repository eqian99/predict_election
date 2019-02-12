import numpy as np
import pandas as pd

df = pd.read_csv('data/train_2008.csv')
X = df[df.columns[:-1].tolist()]
y = df[df.columns[-1]]

test = pd.concat([df['GESTFIPS'], y], axis=1)
test2 = test[test['target'] != 0]
test2 = test2.drop(columns=['target'])
test = test.drop(columns=['target'])
tes2 = test2['GESTFIPS'].value_counts()
tes1 = test['GESTFIPS'].value_counts()
res = (tes2 / tes1)

for i in range(60):
    if i in res.keys():
        df.loc[df['GESTFIPS'] == i, 'GESTFIPS'] = res[i]
    else:
        df.loc[df['GESTFIPS'] == i, 'GESTFIPS'] = res.mean()

print(df['GESTFIPS'])
