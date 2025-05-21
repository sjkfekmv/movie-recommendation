import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import re

# 加载数据集
users_df = pd.read_csv('ml-1m/users.dat', sep='::', names=['user_id', 'gender', 'age', 'occupation', 'zip'], engine='python')
movies_df = pd.read_csv('ml-1m/movies.dat', sep='::', names=['movie_id', 'title', 'genres'], engine='python', encoding='latin-1')
ratings_df = pd.read_csv('ml-1m/ratings.dat', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')

# 预处理：移除电影标题中的年份
def remove_year(title):
    pattern = re.compile(r'^(.*)\((\d+)\)$')
    match = pattern.search(title)
    if match:
        return match.group(1).strip()
    else:
        return title

movies_df['title'] = movies_df['title'].apply(remove_year)

# 预处理：编码性别
le = LabelEncoder()
users_df['gender'] = le.fit_transform(users_df['gender'])

# 检查重复
print("Duplicates in users_df:", users_df.duplicated().sum())
print("Duplicates in movies_df:", movies_df.duplicated().sum())
print("Duplicates in ratings_df:", ratings_df.duplicated().sum())

# 检查异常值：年龄分布
print(users_df['age'].describe())
sns.boxplot(x=users_df['age'])
plt.title('Age Distribution')
plt.savefig('age_distribution.png')
plt.close()

# 可视化：评分频率
sns.countplot(x='rating', data=ratings_df)
plt.title('Frequency of Ratings')
plt.savefig('rating_frequency.png')
plt.close()

# 可视化：电影评分的冒泡图
movie_stats = ratings_df.groupby('movie_id')['rating'].agg(['mean', 'count']).reset_index()
movie_stats = movie_stats[movie_stats['count'] >= 50].nlargest(200, 'count')
movie_stats = movie_stats.merge(movies_df, on='movie_id')
plt.figure(figsize=(12, 8))
plt.scatter(movie_stats['mean'], movie_stats['count'], s=movie_stats['count']/5, alpha=0.5)
plt.title('Movie Ratings: Mean vs Count (Top 200)')
plt.xlabel('Mean Rating')
plt.ylabel('Rating Count')
plt.savefig('movie_ratings_bubble.png')
plt.close()

# 可视化：用户评分的冒泡图
user_stats = ratings_df.groupby('user_id')['rating'].agg(['mean', 'count']).reset_index()
user_stats = user_stats[user_stats['count'] >= 50].nlargest(100, 'count')
user_stats = user_stats.merge(users_df, on='user_id')
plt.figure(figsize=(12, 8))
plt.scatter(user_stats['mean'], user_stats['count'], s=user_stats['count']/5, alpha=0.5)
plt.title('User Ratings: Mean vs Count (Top 100)')
plt.xlabel('Mean Rating')
plt.ylabel('Rating Count')
plt.savefig('user_ratings_bubble.png')
plt.close()

# 职业映射
occupation_map = {
    0: "other or not specified", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
    4: "college/grad student", 5: "customer service", 6: "doctor/health care", 7: "executive/managerial",
    8: "farmer", 9: "homemaker", 10: "K-12 student", 11: "lawyer", 12: "programmer", 13: "retired",
    14: "sales/marketing", 15: "scientist", 16: "self-employed", 17: "technician/engineer",
    18: "tradesman/craftsman", 19: "unemployed", 20: "writer"
}
users_df['occupation_name'] = users_df['occupation'].map(occupation_map)

# 可视化：按职业的用户计数
occupation_counts = users_df['occupation_name'].value_counts()
occupation_counts.plot(kind='bar', figsize=(12, 6))
plt.title('User Counts by Occupation')
plt.xlabel('Occupation')
plt.ylabel('Count')
plt.savefig('occupation_counts.png')
plt.close()

# 年龄组映射
age_map = {
    1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"
}
users_df['age_group'] = users_df['age'].map(age_map)

# 可视化：按年龄组的用户计数
age_counts = users_df['age_group'].value_counts().sort_index()
age_counts.plot(kind='bar', figsize=(10, 6))
plt.title('User Counts by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.savefig('age_counts.png')
plt.close()

# 可视化：性别分布
gender_counts = users_df['gender'].value_counts()
plt.pie(gender_counts, labels=['Male', 'Female'], autopct='%1.1f%%', startangle=90)
plt.title('Gender Distribution')
plt.savefig('gender_distribution.png')
plt.close()

# 可视化：按职业的电影类型比例
movies_df['genres'] = movies_df['genres'].str.split('|')
movies_exploded = movies_df.explode('genres')
ratings_with_genre = ratings_df.merge(movies_exploded[['movie_id', 'genres']], on='movie_id')
ratings_with_user = ratings_with_genre.merge(users_df[['user_id', 'occupation_name']], on='user_id')
occupation_genre_counts = ratings_with_user.groupby(['occupation_name', 'genres']).size().unstack(fill_value=0)
occupation_genre_counts.plot(kind='bar', stacked=True, figsize=(15, 10))
plt.title('Genre Proportions by Occupation')
plt.xlabel('Occupation')
plt.ylabel('Number of Ratings')
plt.savefig('occupation_genre_proportions.png')
plt.close()

# 可视化：每种类型的电影数量
genre_counts = movies_exploded['genres'].value_counts()
genre_counts.plot(kind='bar', figsize=(12, 6))
plt.title('Number of Movies per Genre')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.savefig('genre_counts.png')
plt.close()

# 可视化：按性别的电影类型比例
ratings_with_user = ratings_with_user.merge(users_df[['user_id', 'gender']], on='user_id')
gender_genre_counts = ratings_with_user.groupby(['gender', 'genres']).size().unstack(fill_value=0)
gender_genre_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Genre Proportions by Gender')
plt.xlabel('Gender (0: Male, 1: Female)')
plt.ylabel('Number of Ratings')
plt.savefig('gender_genre_proportions.png')
plt.close()