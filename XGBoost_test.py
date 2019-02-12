from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score
import csv

df = pd.read_csv('data/train_2008.csv')
scaler = StandardScaler()
df[df.columns[:-1].tolist()] = scaler.fit_transform(df[df.columns[:-1].tolist()])
X_train = df[df.columns[:-1].tolist()]
y_train = df[df.columns[-1]]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

df2 = pd.read_csv('data/test_2008.csv')
scaler = StandardScaler()
df2[df2.columns[:-1].tolist()] = scaler.fit_transform(df2[df2.columns[:-1].tolist()])
X_test = df2[df2.columns[:].tolist()]

alg = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5,
                        min_child_weight=3, gamma=0.2, subsample=0.6, colsample_bytree=1.0,
                        objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)

alg.fit(X_train, y_train)
pred_prob = alg.predict_prob(X_test)

with open('output1.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['sep=,'])
    writer.writerow(['id,target'])
    for i in range(len(pred_prob)):
        writer.writerow([str(i) + ',' + str(pred_prob[i])])
