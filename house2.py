#数据的读取
import numpy as np
import pandas as pd
#一般来说源数据的index那一栏没什么用，我们可以用来作为我们pandas dataframe的index。这样之后要是检索起来也省事儿。
train_df = pd.read_csv('./train.csv', index_col=0)
test_df = pd.read_csv('./test.csv', index_col=0)
# test=pd.read_csv('./train.csv')
train_df.head()


prices = pd.DataFrame({"price":train_df["SalePrice"], "log(price + 1)":np.log1p(train_df["SalePrice"])})
prices.hist()

#将训练目标单独拿出
#y_train则是SalePrice那一列
y_train = np.log1p(train_df.pop('SalePrice'))
#把剩下的部分合并起来
all_df = pd.concat((train_df, test_df), axis=0)
all_df.shape
y_train.head()

all_df['MSSubClass'].dtypes
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)
#变成str以后，做个统计
all_df['MSSubClass'].value_counts()

#MSSubClass被分成了12个column，每一个代表一个类。是就是1，不是就是0。
pd.get_dummies(all_df['MSSubClass'], prefix='MSSubClass').head()

#同理，我们把所有的类数据，都给One-Hot了
all_dummy_df = pd.get_dummies(all_df)
all_dummy_df.head()
all_dummy_df.isnull().sum().sort_values(ascending=False).head(10)

#计算平均值
mean_cols = all_dummy_df.mean()
mean_cols.head(10)

#用平均值填补缺失值
all_dummy_df = all_dummy_df.fillna(mean_cols)
#查看填补后是否还有缺失值
all_dummy_df.isnull().sum()#.sum()

#查看哪些数据是数值型的
numeric_cols = all_df.columns[all_df.dtypes != 'object']
numeric_cols

numeric_col_means = all_dummy_df.loc[:, numeric_cols].mean()
numeric_col_std = all_dummy_df.loc[:, numeric_cols].std()
all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std

#将数据分回训练集和测试集
dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]
#查看训练集和测试集的维度
dummy_train_df.shape, dummy_test_df.shape
#首先选用Ridge Regression模型观察效果
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
#把数据框形式转化为numpy Array形式，跟sklearn更配
X_train = dummy_train_df.values
X_test = dummy_test_df.values
#用sklearn自带的cross validation进行模型调参
alphas = np.logspace(-3, 2, 50)
test_scores = []
for alpha in alphas:
    clf = Ridge(alpha)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
#调参结果可视化
import matplotlib.pyplot as plt
# %matplotlib inline
plt.plot(alphas, test_scores)
plt.title("Alpha vs CV Error")

from sklearn.ensemble import RandomForestRegressor
max_features = [.1, .3, .5, .7, .9, .99]
test_scores = []
for max_feat in max_features:
    clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
plt.plot(max_features, test_scores)
plt.title("Max Features vs CV Error");

#在这里取alpha=15
from sklearn.linear_model import Ridge
ridge = Ridge(15)
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score
#在这里，我们用CV结果来测试不同的分类器个数对最后结果的影响。
#注意，我们在用Bagging的时候，要把它的函数base_estimator里填上小分类器（ridge）
params = [1, 10, 15, 20, 25, 30, 40]
test_scores = []
for param in params:
    clf = BaggingRegressor(n_estimators=param, base_estimator=ridge)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
import matplotlib.pyplot as plt
# %matplotlib inline
plt.plot(params, test_scores)
plt.title("n_estimator vs CV Error")

from xgboost import XGBRegressor
params = [1,2,3,4,5,6]
test_scores = []
for param in params:
    clf = XGBRegressor(max_depth=param)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
import matplotlib.pyplot as plt
# %matplotlib inline
plt.plot(params, test_scores)
plt.title("max_depth vs CV Error");
