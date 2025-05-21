import pandas as pd
import numpy as np
import torch
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 检查是否可以使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载函数（保持不变）
def load_data():
    users_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']
    users = pd.read_csv('./ml-1m/users.dat', sep='::', header=None, names=users_title, engine='python', encoding='latin-1')
    users = users.filter(regex='UserID')
    
    movies_title = ['MovieID', 'Title', 'Genres']
    movies = pd.read_csv('./ml-1m/movies.dat', sep='::', header=None, names=movies_title, engine='python', encoding='latin-1')
    
    ratings_title = ['UserID', 'MovieID', 'Rating', 'timestamps']
    ratings = pd.read_csv('./ml-1m/ratings.dat', sep='::', header=None, names=ratings_title, engine='python', encoding='latin-1')
    
    data = pd.merge(pd.merge(ratings, users), movies)
    
    movieid2idx = {val[0]: i for i, val in enumerate(movies.values)}
    
    return ratings, users, movies, data, movieid2idx

# 推荐召回率指标（保持不变）
class RecommendationMetrics:
    @staticmethod
    def recall_at_k(recommended_items, relevant_items, k=10):
        if len(relevant_items) == 0:
            return 1
        count = 0
        for item in recommended_items[:k]:
            if item in relevant_items:
                count += 1
        return count / (1.0 * len(relevant_items))

# 修改后的推荐函数：仅对用户在测试集中交互过的电影进行预测
def recommend_movies_for_user(user_id, test_data, users_matrics, movie_matrics, movieid2idx, k=10):
    # 获取用户在测试集中交互过的电影
    user_test_data = test_data[test_data['UserID'] == user_id]
    test_movie_ids = user_test_data['MovieID'].values
    
    # 过滤出有效的电影索引（确保电影ID存在于movieid2idx中）
    test_movie_indices = [movieid2idx[movie_id] for movie_id in test_movie_ids if movie_id in movieid2idx]
    if len(test_movie_indices) == 0:
        return []  # 如果没有有效的电影索引，返回空列表
    
    # 获取用户特征和测试集电影特征
    user_features = torch.tensor(users_matrics[user_id-1], dtype=torch.float).view(1, -1).to(device)
    movie_features = torch.tensor(movie_matrics[test_movie_indices], dtype=torch.float).to(device)
    
    # 计算评分
    scores = torch.matmul(user_features, movie_features.t())
    
    # 排序并获取前K个电影的索引（对应于test_movie_indices的相对索引）
    top_relative_indices = np.argsort(scores.cpu().numpy().squeeze())[-k:][::-1]
    top_movie_indices = [test_movie_indices[i] for i in top_relative_indices]
    
    return top_movie_indices

# 获取用户喜欢的电影（保持不变）
def get_actual_liked_movies(user_id, data, threshold=4):
    user_ratings = data[data['UserID'] == user_id]
    liked_movies = user_ratings[user_ratings['Rating'] >= threshold]['MovieID'].values
    liked_movies=sorted(liked_movies)
    return liked_movies[:10]

# 修改后的主评估函数
def evaluate_recommendation_system():
    print("加载数据...")
    ratings, users, movies, data, movieid2idx = load_data()
    
    # 按时间点划分测试集
    min_ratings = 10
    user_rating_counts = ratings.groupby('UserID').size()
    valid_users = user_rating_counts[user_rating_counts >= min_ratings].index
    valid_ratings = ratings[ratings['UserID'].isin(valid_users)]
    
    # 按时间戳排序
    valid_ratings = valid_ratings.sort_values(by='timestamps')
    
    # 按时间点划分，最后 20% 的评分作为测试集
    test_size = 0.2
    split_idx = int(len(valid_ratings) * (1 - test_size))
    train_ratings = valid_ratings.iloc[:split_idx]
    test_ratings = valid_ratings.iloc[split_idx:]
    print(f"训练集大小: {len(train_ratings)}, 测试集大小: {len(test_ratings)}")
    
    # 确保测试集中的用户在训练集中有评分
    train_users = train_ratings['UserID'].unique()
    test_ratings = test_ratings[test_ratings['UserID'].isin(train_users)]
    test_data = pd.merge(pd.merge(test_ratings, users), movies)
    
    # 加载特征矩阵
    print("\n=== 加载特征矩阵 ===")
    try:
        movie_matrics = pickle.load(open('movie_matrics_new.p', 'rb'))
        users_matrics = pickle.load(open('users_matrics_new.p', 'rb'))
    except:
        print("特征矩阵未找到。请先生成 movie_matrics.p 和 users_matrics.p。")
        return
    
    # 评估推荐召回率
    print("\n=== 评估推荐召回率 ===")
    min_liked_movies = 5
    valid_users = []
    for user_id in test_data['UserID'].unique():
        liked_movies = get_actual_liked_movies(user_id, test_data)
        if len(liked_movies) >= min_liked_movies:
            valid_users.append(user_id)
    
    sample_users = np.random.choice(valid_users, min(100, len(valid_users)), replace=False)
    
    avg_recall = 0
    user_count = 0
    
    for user_id in sample_users:
        # 获取用户在测试集中喜欢的电影（ground truth）
        liked_movies = get_actual_liked_movies(user_id, test_data)
        relevant_indices = [movieid2idx[movie_id] for movie_id in liked_movies if movie_id in movieid2idx]
        
        # 获取推荐电影（仅基于测试集中交互过的电影）
        recommended_indices = recommend_movies_for_user(user_id, test_data, users_matrics, movie_matrics, movieid2idx)
        
        if len(relevant_indices) > 0 and len(recommended_indices) > 0:
            recall = RecommendationMetrics.recall_at_k(recommended_indices, set(relevant_indices))
            avg_recall += recall
            user_count += 1
    
    if user_count > 0:
        avg_recall /= user_count
        print(f"推荐召回率@10: {avg_recall:.4f}")
    else:
        print("没有足够的数据来计算召回率。")

if __name__ == "__main__":
    evaluate_recommendation_system()
    print("评估完成。")