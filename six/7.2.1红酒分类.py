# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 08:53:45 2024

@author: Administrator
"""
# 导入科学计算库
import numpy as np
# 导入绘图库
import matplotlib.pyplot as plt
# 导入决策树模块
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
# 导入红酒数据库
from sklearn.datasets import load_wine
# 导入数据划分模块
from sklearn.model_selection import train_test_split
# 导入数据标准化模块
from sklearn.preprocessing import StandardScaler
# 导入混淆矩阵
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 加载红酒数据
wine = load_wine()
print('数据基本信息:', wine.data.shape)
print('特征名称：', wine.feature_names)
print('类别名称：', wine.target_names)

# 分离特征与类别标记
x = wine.data
y = wine.target

# 数据标准化
scaler = StandardScaler()
x_ = scaler.fit_transform(x)

# 将数据划分为训练数据与测试数据
x_train, x_test, y_train, y_test = train_test_split(x_, y, test_size=0.3, random_state=10)

# 1、特征工程 - 选取最佳的5个特征
print("\n1. 特征工程 - 选取最佳的5个特征")

# 使用决策树计算特征重要性
dt_feature_selector = DecisionTreeClassifier(max_depth=4, random_state=10)
dt_feature_selector.fit(x_train, y_train)
feature_importance = dt_feature_selector.feature_importances_

# 获取重要性排名前5的特征
top5_idx = np.argsort(feature_importance)[-5:][::-1]
top5_features = np.array(wine.feature_names)[top5_idx]
top5_importance = feature_importance[top5_idx]

print("前5个重要特征及其重要性:")
for i, (feature, importance) in enumerate(zip(top5_features, top5_importance)):
    print(f"{i+1}. {feature}: {importance:.4f}")

# 使用前5个重要特征
x_train_top5 = x_train[:, top5_idx]
x_test_top5 = x_test[:, top5_idx]

# 2、基于决策树模型进行训练和可视化
print("\n2. 构建决策树模型（深度=4，基尼系数）")

# 构建决策树模型，使用基尼系数作为分类准则
DT = DecisionTreeClassifier(max_depth=4, criterion='gini', random_state=10)
DT.fit(x_train_top5, y_train)

# 绘制决策树
plt.figure(figsize=(20, 10))
tree.plot_tree(DT,
               feature_names=top5_features.tolist(),
               class_names=wine.target_names.tolist(),
               filled=True,
               rounded=True,
               fontsize=10)
plt.title('Decision Tree for Wine Classification (Depth=4, Gini Criterion)', fontsize=16)
plt.show()

# 3、测试集评估和混淆矩阵
print("\n3. 测试集评估")

# 在测试集上进行预测
y_pred = DT.predict(x_test_top5)

# 计算准确率
train_accuracy = DT.score(x_train_top5, y_train)
test_accuracy = DT.score(x_test_top5, y_test)

print(f"训练集准确率: {train_accuracy:.4f}")
print(f"测试集准确率: {test_accuracy:.4f}")

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=wine.target_names)

plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Wine Classification', fontsize=14)
plt.show()

# 打印详细的混淆矩阵
print("\n混淆矩阵:")
print("行: 真实类别, 列: 预测类别")
print(" " * 10, end="")
for name in wine.target_names:
    print(f"{name:^12}", end="")
print()

for i, true_name in enumerate(wine.target_names):
    print(f"{true_name:10}", end="")
    for j in range(len(wine.target_names)):
        print(f"{cm[i, j]:^12}", end="")
    print()

# 额外：显示特征重要性图
plt.figure(figsize=(10, 6))
plt.barh(range(len(top5_features)), top5_importance, align='center')
plt.yticks(range(len(top5_features)), top5_features)
plt.xlabel('Feature Importance')
plt.title('Top 5 Feature Importance')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()