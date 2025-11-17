# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:54:03 2024

@author: Administrator
"""

# 导入科学计算库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # 导入绘图库
from sklearn.metrics import mean_squared_error  # 导入均方误差模块
from sklearn.model_selection import train_test_split  # 导入数据划分模块
from sklearn.linear_model import LinearRegression  # 导入线性回归库
from sklearn import preprocessing  # 导入数据预处理模块

# 加载波士顿房价的数据集
data_url = "http://lib.stat.cmu.edu/datasets/boston"  # 数据网址
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)  # 读取数据
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
boston_df = pd.DataFrame(data,
                         columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
                                  'B', 'LSTAT'])

boston_y = pd.DataFrame(target, columns=['MEDV'])  # 将数据转化成DataFrame格式
boston_df['MEDV'] = boston_y  # 新增一列
boston_df.head()  # 显示数据前5行
boston_df.describe()  # 查看数据的描述信息

# 计算每一个特征和房价的相关系数，并选择最相关的5个特征
corr_with_medv = boston_df.corr()['MEDV'].abs().sort_values(ascending=False)
top_5_features = corr_with_medv[1:6].index.tolist()  # 排除MEDV自身，取前5个

print("最相关的5个特征:")
for i, feature in enumerate(top_5_features, 1):
    print(f"{i}. {feature}: {corr_with_medv[feature]:.4f}")

# 数据处理 - 选择最相关的5个特征
X = np.array(boston_df[top_5_features])  # 特征值
y = np.array(boston_df['MEDV'])  # 目标值

# 划分测试集与训练集 (80%训练, 20%测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# 归一化
min_max_scaler = preprocessing.MinMaxScaler()
# 分别对训练和测试数据的特征以及目标值进行标准化处理
X_train = min_max_scaler.fit_transform(X_train)
y_train = min_max_scaler.fit_transform(y_train.reshape(-1, 1))  # reshape(-1,1)指将它转化为1列
X_test = min_max_scaler.fit_transform(X_test)
y_test = min_max_scaler.fit_transform(y_test.reshape(-1, 1))

# 线性回归
lr = LinearRegression()  # 建立线性回归模型
lr.fit(X_train, y_train)  # 使用训练数据进行参数估计

# 输出斜率(系数)和截距
print("\n线性回归模型参数:")
print("截距 (intercept):", lr.intercept_[0])
print("斜率 (coefficients):")
for i, feature in enumerate(top_5_features):
    print(f"  {feature}: {lr.coef_[0][i]:.6f}")

# 使用测试数据进行回归预测
y_test_pred = lr.predict(X_test)

# 利用测试样本测试模型的精度
acc = lr.score(X_test, y_test)
# 计算均方误差
mse = mean_squared_error(y_test, y_test_pred)

# 输出模型精度和均方误差
print(f"\n模型评估:")
print('精度 (R² score):', acc)
print('均方误差 (MSE):', mse)

# 分别绘制这5个特征和房价之间关系的折线图
plt.figure(figsize=(15, 10))
for i, feature in enumerate(top_5_features):
    plt.subplot(2, 3, i + 1)

    # 对单个特征进行线性回归
    single_lr = LinearRegression()
    single_lr.fit(X_train[:, i].reshape(-1, 1), y_train)
    single_acc = single_lr.score(X_test[:, i].reshape(-1, 1), y_test)

    # 绘制散点图
    plt.scatter(X_train[:, i], y_train, s=20, c='green', edgecolor='black', alpha=0.6, label='训练数据')
    plt.scatter(X_test[:, i], y_test, s=20, c='blue', edgecolor='black', alpha=0.6, label='测试数据')

    # 绘制回归线
    x_range = np.linspace(X[:, i].min(), X[:, i].max(), 100).reshape(-1, 1)
    x_range_scaled = min_max_scaler.fit_transform(x_range)
    y_pred_range = single_lr.predict(x_range_scaled)

    plt.plot(x_range_scaled, y_pred_range, c='red', linewidth=2, label='回归线')

    plt.xlabel(feature)
    plt.ylabel('MEDV (标准化)')
    plt.title(f'{feature} vs MEDV\nAccuracy: {single_acc:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

import matplotlib.pyplot as plt
# 明确设置字体为 SimHei
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False
# 然后进行绘图等操作
plt.tight_layout()
plt.show()

# 绘制所有特征的相关系数条形图
plt.figure(figsize=(12, 6))
corr_with_medv[1:].sort_values().plot.bar(color='skyblue')
plt.title('所有特征与房价的相关系数')
plt.ylabel('Correlation with MEDV')
plt.xlabel('Features')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()