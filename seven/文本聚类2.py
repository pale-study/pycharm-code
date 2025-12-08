from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from collections import Counter
from sklearn.metrics import silhouette_score
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    """加载数据，处理网络问题"""
    try:
        # 尝试使用本地缓存
        categories = [
            'rec.autos', 'comp.graphics', 'sci.space',
            'talk.politics.mideast', 'rec.sport.baseball', 'sci.med'
        ]

        newsgroups = fetch_20newsgroups(
            subset='train',
            categories=categories,
            shuffle=True,
            random_state=42,
            remove=('headers', 'footers', 'quotes'),
            download_if_missing=False  # 不自动下载
        )
        return newsgroups

    except Exception as e:
        print(f"加载数据: {e}")
        print("使用模拟数据进行演示...")
        return create_sample_data()


def create_sample_data():
    """创建模拟数据用于演示"""
    from sklearn.datasets import make_classification
    from sklearn.utils import Bunch

    # 创建模拟文本数据
    sample_texts = [
                       "car engine drive road vehicle auto speed",
                       "computer graphics image design software digital",
                       "space nasa earth orbit moon satellite",
                       "government political war country policy middle east",
                       "baseball game team player sport score",
                       "medical health doctor patient disease hospital",
                       "vehicle truck highway transport traffic",
                       "software program code computer system",
                       "planet mars universe science astronomy",
                       "election president political nation",
                       "basketball football sport competition",
                       "medicine treatment drug health care"
                   ] * 50  # 复制创建更多样本

    # 创建对应的标签
    sample_target = np.array([0, 1, 2, 3, 4, 5] * 100)

    return Bunch(
        data=sample_texts,
        target=sample_target,
        target_names=['autos', 'graphics', 'space', 'politics', 'sports', 'medical']
    )


# 加载数据
print("正在加载数据...")
newsgroups = load_data()

print(f"文档数量: {len(newsgroups.data)}")
print(f"类别: {newsgroups.target_names}")
print(f"第一个文档: {newsgroups.data[0][:100]}...")

# 文本向量化
vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    min_df=2,
    max_df=0.8
)

X = vectorizer.fit_transform(newsgroups.data)
print(f"向量化后的矩阵形状: {X.shape}")

# 任务1: 针对所有特征确定最佳K值
print("=" * 50)
print("任务1: 针对所有特征确定最佳K值")
print("=" * 50)

inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, clusters))

# 绘制图表
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('簇数量 (K)')
plt.ylabel('簇内平方和')
plt.title('肘部法则 - 所有特征')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(k_range, silhouette_scores, 'ro-')
plt.xlabel('簇数量 (K)')
plt.ylabel('轮廓系数')
plt.title('轮廓系数 - 所有特征')
plt.grid(True)

best_k_silhouette = k_range[np.argmax(silhouette_scores)]
print(f"根据轮廓系数建议的最佳K值: {best_k_silhouette}")

# 使用最佳K值聚类
kmeans_all = KMeans(n_clusters=best_k_silhouette, random_state=42, n_init=10)
clusters_all = kmeans_all.fit_predict(X)

# 可视化
pca_all = PCA(n_components=2, random_state=42)
X_2d_all = pca_all.fit_transform(X.toarray())

plt.subplot(1, 3, 3)
scatter = plt.scatter(X_2d_all[:, 0], X_2d_all[:, 1], c=clusters_all, cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.title(f'聚类结果 (K={best_k_silhouette})')
plt.tight_layout()
plt.show()

# 任务2: PCA降维到500维
print("\n" + "=" * 50)
print("任务2: PCA降维到500维后确定最佳K值")
print("=" * 50)

# 如果特征数少于500，使用所有特征
n_components = min(500, X.shape[1])
pca_500 = PCA(n_components=n_components, random_state=42)
X_pca = pca_500.fit_transform(X.toarray())
print(f"PCA降维后的矩阵形状: {X_pca.shape}")

inertia_pca = []
silhouette_scores_pca = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)
    inertia_pca.append(kmeans.inertia_)
    silhouette_scores_pca.append(silhouette_score(X_pca, clusters))

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(k_range, inertia_pca, 'bo-')
plt.xlabel('簇数量 (K)')
plt.ylabel('簇内平方和')
plt.title('肘部法则 - PCA降维后')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(k_range, silhouette_scores_pca, 'ro-')
plt.xlabel('簇数量 (K)')
plt.ylabel('轮廓系数')
plt.title('轮廓系数 - PCA降维后')
plt.grid(True)

best_k_pca = k_range[np.argmax(silhouette_scores_pca)]
print(f"PCA降维后根据轮廓系数建议的最佳K值: {best_k_pca}")

kmeans_pca = KMeans(n_clusters=best_k_pca, random_state=42, n_init=10)
clusters_pca = kmeans_pca.fit_predict(X_pca)

pca_2d = PCA(n_components=2, random_state=42)
X_2d_pca = pca_2d.fit_transform(X_pca)

plt.subplot(1, 3, 3)
scatter = plt.scatter(X_2d_pca[:, 0], X_2d_pca[:, 1], c=clusters_pca, cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.title(f'PCA降维后聚类 (K={best_k_pca})')
plt.tight_layout()
plt.show()

# 任务3: 输出轮廓系数
print("\n" + "=" * 50)
print("任务3: 输出降维后的平均轮廓系数")
print("=" * 50)

silhouette_avg_pca = silhouette_score(X_pca, clusters_pca)
silhouette_avg_original = silhouette_score(X, clusters_all)

print(f"原始特征的平均轮廓系数: {silhouette_avg_original:.3f}")
print(f"PCA降维后的平均轮廓系数: {silhouette_avg_pca:.3f}")
print(f"轮廓系数变化: {silhouette_avg_pca - silhouette_avg_original:+.3f}")

# 总结
print("\n" + "=" * 50)
print("总结")
print("=" * 50)
print(f"1. 原始特征最佳K值: {best_k_silhouette}, 轮廓系数: {silhouette_avg_original:.3f}")
print(f"2. PCA降维后最佳K值: {best_k_pca}, 轮廓系数: {silhouette_avg_pca:.3f}")