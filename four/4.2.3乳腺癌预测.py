# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 17:13:26 2024

@author: Administrator
"""
import matplotlib.pyplot as plt  # 导入绘图库
import numpy as np  # 导入科学计算库
from sklearn.datasets import load_breast_cancer  # 导入数据库
from sklearn.model_selection import train_test_split  # 导入样本集划分模块
from sklearn import preprocessing  # 导入数据预处理库
from sklearn.linear_model import LogisticRegression  # 导入Logistic回归库
from sklearn.model_selection import cross_val_score  # 导入交叉验证库
from sklearn.feature_selection import SelectFromModel  # 导入特征选择库




# 加载数据
Cancer = load_breast_cancer()
x = Cancer.data  # 特征值
y = Cancer.target  # 目标值

# 输出数据基本信息
print('数据基本信息: {0}; Cancer_No: {1}; Cancer_Yes: {2}'.format(x.shape, y[y == 1].shape[0], y[y == 0].shape[0]))
print('特征名称:', Cancer.feature_names)

# 将样本集划分为训练样本与测试样本
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)
# 对数据进行标准化处理
transfer = preprocessing.StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 构建逻辑回归模型
LR = LogisticRegression()
LR.fit(x_train, y_train)

# 模型评估
print('预测精度:', LR.score(x_test, y_test))

# 在L1正则化的基础上通过遍历C值的方式确定特征提取前与特征提取后的精度
all_features = []
selected_features = []
selected_feature_counts = []  # 存储特征选择后的特征数量
C = np.arange(0.01, 10, 0.5)

for i in C:
    # 构建Logistic回归模型 - 修改为L1正则化
    LR = LogisticRegression(penalty='l1', solver="liblinear", C=i, random_state=100)

    # 特征提取前交叉验证精度
    all_features.append(cross_val_score(LR, x_train, y_train, cv=10).mean())

    # 特征提取后交叉验证精度
    selector = SelectFromModel(LR, norm_order=1).fit(x_train, y_train)
    X_new_train = selector.transform(x_train)
    selected_features.append(cross_val_score(LR, X_new_train, y_train, cv=10).mean())

    # 记录特征选择后的特征数量
    selected_feature_counts.append(X_new_train.shape[1])

# 输出特征提取前模型精度最高值及对应的C值
print('特征提取前模型精度最高值及对应的C值:', max(all_features), C[all_features.index(max(all_features))])

# 输出特征提取后模型精度最高值及对应的C值
max_selected_idx = selected_features.index(max(selected_features))
print('特征提取后模型精度最高值及对应的C值:', max(selected_features), C[max_selected_idx])
print('特征选择后的特征数量:', selected_feature_counts[max_selected_idx])

# 获取最佳C值对应的特征选择器，用于后续分析
best_C = C[max_selected_idx]
LR_best = LogisticRegression(penalty='l1', solver="liblinear", C=best_C, random_state=100)
selector_best = SelectFromModel(LR_best, norm_order=1).fit(x_train, y_train)
X_new_train_best = selector_best.transform(x_train)

# 获取被选择的特征名称
selected_feature_indices = selector_best.get_support()
selected_feature_names = [Cancer.feature_names[i] for i in range(len(selected_feature_indices)) if
                          selected_feature_indices[i]]
print(f'特征选择后保留的特征数量: {len(selected_feature_names)}')
print('被选择的特征名称:', selected_feature_names)

plt.figure(figsize=(12, 6))
plt.plot(C, all_features, label="All Features")
plt.plot(C, selected_features, label="Selected Features")
plt.xticks(C)
plt.grid(True)
plt.legend()
plt.xlabel("C Value")
plt.ylabel("Accuracy")

# 在图像上添加特征数量信息
selected_text = f'Selected Features: {len(selected_feature_names)}\n'
for i, feature in enumerate(selected_feature_names):
    if i < 8:  # 限制显示的特征数量，避免图像过于拥挤
        selected_text += f'{feature}\n'
    else:
        selected_text += '...\n'
        break

plt.text(0.02, 0.98, selected_text, transform=plt.gca().transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         fontsize=9)

# 添加特征数量变化的注释
for i, count in enumerate(selected_feature_counts):
    if i % 5 == 0:  # 每5个点标注一次，避免过于密集
        plt.annotate(f'{count}', (C[i], selected_features[i]),
                     textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

plt.title('L1 Regularization - Feature Selection Performance')
plt.tight_layout()
plt.show()

# 构建与训练Logistic回归模型 - 使用L1正则化
LR = LogisticRegression(penalty='l1', solver='liblinear', C=best_C, random_state=100)
x_new = SelectFromModel(LR, norm_order=1).fit(x_train, y_train)
x_new_train = x_new.transform(x_train)
LR.fit(x_new_train, y_train)

# 利用训练数据测试Logistic回归模型的精度
print('训练数据相应的精度:', cross_val_score(LR, x_new_train, y_train, cv=10).mean())

# 利用测试数据测试Logistic回归模型的精度
x_new_test = x_new.transform(x_test)
print('测试数据相应的精度:', cross_val_score(LR, x_new_test, y_test, cv=10).mean())

# 输出详细的特征选择信息
print(f'\n=== 特征选择详细信息 ===')
print(f'原始特征数量: {x_train.shape[1]}')
print(f'特征选择后保留的特征数量: {x_new_train.shape[1]}')
print(f'特征减少比例: {(1 - x_new_train.shape[1] / x_train.shape[1]) * 100:.2f}%')
print('被选择的特征:')
for i, feature in enumerate(selected_feature_names, 1):
    print(f'{i}. {feature}')

    # 修改样本划分，确保20%测试数据
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=22
    )

    # 添加测试集精度评估
    print('测试集精度（特征选择后）:', LR.score(x_new_test, y_test))