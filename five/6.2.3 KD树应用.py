# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler


def create_sample_data():
    """创建示例电影数据"""
    # 示例电影数据
    sample_movies = [
        # Titanic 相似电影（浪漫/剧情）
        [1, 'Titanic (1997)', 'Drama|Romance', 4.1, 500],
        [2, 'The Notebook (2004)', 'Drama|Romance', 4.0, 450],
        [3, 'A Walk to Remember (2002)', 'Drama|Romance', 3.8, 300],
        [4, 'Romeo + Juliet (1996)', 'Drama|Romance', 3.9, 350],
        [5, 'Pearl Harbor (2001)', 'Action|Drama|Romance', 3.7, 400],

        # Saving Private Ryan 相似电影（战争/动作）
        [6, 'Saving Private Ryan (1998)', 'Action|Drama|War', 4.3, 600],
        [7, 'Black Hawk Down (2001)', 'Action|Drama|War', 4.0, 350],
        [8, 'The Thin Red Line (1998)', 'Drama|War', 3.8, 250],
        [9, 'Platoon (1986)', 'Drama|War', 4.2, 300],
        [10, 'Full Metal Jacket (1987)', 'Drama|War', 4.1, 320],

        # Roman Holiday 相似电影（浪漫/喜剧）
        [11, 'Roman Holiday (1953)', 'Comedy|Romance', 4.4, 200],
        [12, 'Sabrina (1954)', 'Comedy|Romance', 4.0, 150],
        [13, 'Funny Face (1957)', 'Comedy|Musical|Romance', 3.9, 120],
        [14, 'Breakfast at Tiffany\'s (1961)', 'Comedy|Drama|Romance', 4.2, 280],
        [15, 'An Affair to Remember (1957)', 'Drama|Romance', 4.1, 180],

        # 其他电影增加多样性
        [16, 'The Godfather (1972)', 'Crime|Drama', 4.8, 800],
        [17, 'Pulp Fiction (1994)', 'Crime|Drama', 4.5, 700],
        [18, 'Forrest Gump (1994)', 'Drama|Romance', 4.6, 750],
        [19, 'The Shawshank Redemption (1994)', 'Drama', 4.9, 900],
        [20, 'The Dark Knight (2008)', 'Action|Crime|Drama', 4.7, 850]
    ]

    movies_df = pd.DataFrame(sample_movies,
                             columns=['movieId', 'title', 'genres', 'avg_rating', 'rating_count'])

    return movies_df


# 使用示例数据
movies = create_sample_data()

# 处理类型特征
genres_dummies = movies['genres'].str.get_dummies(sep='|')
feature_columns = list(genres_dummies.columns) + ['avg_rating', 'rating_count']
features = pd.concat([genres_dummies, movies[['avg_rating', 'rating_count']]], axis=1)

# 标准化特征
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 构建 KD 树
tree = KDTree(features_scaled, leaf_size=10, metric='euclidean')


# 定义推荐函数
def recommend_movies(movie_title, k=6):
    """推荐相似电影"""
    if movie_title not in movies['title'].values:
        # 尝试模糊匹配
        matches = movies[movies['title'].str.contains(movie_title, case=False, na=False)]
        if len(matches) == 0:
            print(f"未找到电影: {movie_title}")
            return
        else:
            idx = matches.index[0]
            movie_title = matches.iloc[0]['title']
    else:
        idx = movies.index[movies['title'] == movie_title].tolist()[0]

    point = features_scaled[idx].reshape(1, -1)
    dists, indices = tree.query(point, k=k + 1)  # k+1 因为包含自己

    # 排除自己
    sim_indices = indices[0][1:]
    sim_dists = dists[0][1:]

    print(f"\n基于电影《{movie_title}》推荐的 {k-1} 部电影：")
    print("-" * 50)
    for i, (idx, dist) in enumerate(zip(sim_indices, sim_dists)):
        title = movies.iloc[idx]['title']
        genres = movies.iloc[idx]['genres']
        similarity = 1 / (1 + dist)  # 将距离转换为相似度
        print(f"{i + 1}. {title}")
        print(f"   类型: {genres}")
        print(f"   相似度: {similarity:.4f}")
        print()


# 输出推荐结果
print("KD树电影推荐系统")
print("=" * 50)

recommend_movies('Titanic')
recommend_movies('Saving Private Ryan')
recommend_movies('Roman Holiday')