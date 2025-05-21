import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
import pickle
import re
import random
from tqdm import tqdm

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 1. Data loading and preprocessing
def load_data():
    """
    Load Dataset from File
    """
    # 读取User数据
    users_title = ["UserID", "Gender", "Age", "JobID", "Zip-code"]
    users = pd.read_csv(
        "./ml-1m/users.dat",
        sep="::",
        header=None,
        names=users_title,
        engine="python",
        encoding="latin-1",
    )
    users = users.filter(regex="UserID|Gender|Age|JobID")
    users_orig = users.values
    # 改变User数据中性别和年龄
    gender_map = {"F": 0, "M": 1}
    users["Gender"] = users["Gender"].map(gender_map)

    age_map = {val: ii for ii, val in enumerate(set(users["Age"]))}
    users["Age"] = users["Age"].map(age_map)

    # 读取Movie数据集
    movies_title = ["MovieID", "Title", "Genres"]
    movies = pd.read_csv(
        "./ml-1m/movies.dat",
        sep="::",
        header=None,
        names=movies_title,
        engine="python",
        encoding="latin-1",
    )
    movies_orig = movies.values
    # 将Title中的年份去掉
    pattern = re.compile(r"^(.*)\((\d{4})\)$")

    title_map = {
        val: pattern.match(val).group(1) for ii, val in enumerate(set(movies["Title"]))
    }
    movies["Title"] = movies["Title"].map(title_map)

    # 电影类型转数字字典
    genres_set = set()
    for val in movies["Genres"].str.split("|"):
        genres_set.update(val)

    genres_set.add("<PAD>")
    genres2int = {val: ii for ii, val in enumerate(genres_set)}

    # 将电影类型转成等长数字列表，长度是18
    genres_map = {
        val: [genres2int[row] for row in val.split("|")]
        for ii, val in enumerate(set(movies["Genres"]))
    }

    for key in genres_map:
        for cnt in range(max(genres2int.values()) + 1 - len(genres_map[key])):
            genres_map[key].insert(len(genres_map[key]) + cnt, genres2int["<PAD>"])

    movies["Genres"] = movies["Genres"].map(genres_map)

    # 电影Title转数字字典
    title_set = set()
    for val in movies["Title"].str.split():
        title_set.update(val)

    title_set.add("<PAD>")
    title2int = {val: ii for ii, val in enumerate(title_set)}

    # 将电影Title转成等长数字列表，长度是15
    title_count = 15
    title_map = {
        val: [title2int[row] for row in val.split()]
        for ii, val in enumerate(set(movies["Title"]))
    }

    for key in title_map:
        for cnt in range(title_count - len(title_map[key])):
            title_map[key].insert(len(title_map[key]) + cnt, title2int["<PAD>"])

    movies["Title"] = movies["Title"].map(title_map)

    # 读取评分数据集
    ratings_title = ["UserID", "MovieID", "Rating", "timestamps"]
    ratings = pd.read_csv(
        "./ml-1m/ratings.dat",
        sep="::",
        header=None,
        names=ratings_title,
        engine="python",
        encoding="latin-1",
    )
    ratings = ratings.filter(regex="UserID|MovieID|Rating")

    # 合并三个表
    data = pd.merge(pd.merge(ratings, users), movies)

    # 将数据分成X和y两张表
    target_fields = ["Rating"]
    features_pd, targets_pd = data.drop(target_fields, axis=1), data[target_fields]

    features = features_pd.values
    targets_values = targets_pd.values

    # 创建电影ID到索引的映射
    movieid2idx = {val[0]: i for i, val in enumerate(movies.values)}

    return (
        title_count,
        title_set,
        genres2int,
        features,
        targets_values,
        ratings,
        users,
        movies,
        data,
        movies_orig,
        users_orig,
        movieid2idx,
    )


# 2. Create MovieLensDataset
class MovieLensDataset(Dataset):
    def __init__(
        self, features, targets, movies, title_count=15, poster_dir="./poster/"
    ):
        self.features = features
        self.targets = targets
        self.movies = movies
        self.title_count = title_count
        self.poster_dir = poster_dir
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Extract features
        user_id = int(self.features[idx, 0]) - 1  # 0-indexed
        movie_id = int(self.features[idx, 1])
        gender = int(self.features[idx, 2])
        age = int(self.features[idx, 3])
        job_id = int(self.features[idx, 4])

        # Find movie in the movies dataframe
        movie_idx = self.movies[self.movies["MovieID"] == movie_id].index[0]
        movie_categories = self.movies.iloc[movie_idx]["Genres"]
        movie_titles = self.movies.iloc[movie_idx]["Title"]

        rating = float(self.targets[idx, 0])

        # Convert to tensors
        gender_tensor = torch.tensor(gender, dtype=torch.long)
        age_tensor = torch.tensor(age, dtype=torch.long)
        job_id_tensor = torch.tensor(job_id, dtype=torch.long)
        user_id_tensor = torch.tensor(user_id, dtype=torch.long)
        movie_id_tensor = torch.tensor(movie_id - 1, dtype=torch.long)  # 0-indexed

        # Convert categories and titles to tensors
        movie_categories_tensor = torch.tensor(movie_categories, dtype=torch.long)
        movie_titles_tensor = torch.tensor(movie_titles, dtype=torch.long)

        # Load movie poster image
        poster_path = os.path.join(self.poster_dir, f"{movie_id}.jpg")
        try:
            poster_img = Image.open(poster_path).convert("RGB")
            poster_img = self.transform(poster_img)
        except (FileNotFoundError, IOError):
            # If poster not found, use a placeholder of zeros
            poster_img = torch.zeros(3, 224, 224)

        return {
            "user_id": user_id_tensor,
            "movie_id": movie_id_tensor,
            "gender": gender_tensor,
            "age": age_tensor,
            "job_id": job_id_tensor,
            "movie_categories": movie_categories_tensor,
            "movie_titles": movie_titles_tensor,
            "poster_img": poster_img,
            "rating": torch.tensor(rating, dtype=torch.float),
        }


# 3. Define model architecture
class MovieRecommendationModel(nn.Module):
    def __init__(
        self,
        n_users,
        n_movies,
        n_genders,
        n_ages,
        n_jobs,
        n_categories,
        n_titles,
        embed_dim=32,
        title_dim=15,
    ):
        super(MovieRecommendationModel, self).__init__()
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.movie_embedding = nn.Embedding(n_movies, embed_dim)
        self.gender_embedding = nn.Embedding(n_genders, embed_dim // 2)
        self.age_embedding = nn.Embedding(n_ages, embed_dim // 2)
        self.job_embedding = nn.Embedding(n_jobs, embed_dim // 2)
        self.category_embedding = nn.Embedding(n_categories, embed_dim)
        self.title_embedding = nn.Embedding(n_titles, embed_dim)

        # Poster image feature extractor
        self.poster_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 200),
        )

        # CNN for titles
        self.title_conv = nn.ModuleList(
            [nn.Conv1d(embed_dim, 8, kernel_size=k) for k in [2, 3, 4, 5]]
        )

        # User feature layers
        self.user_fc = nn.Linear(embed_dim, embed_dim)
        self.gender_fc = nn.Linear(embed_dim // 2, embed_dim)
        self.age_fc = nn.Linear(embed_dim // 2, embed_dim)
        self.job_fc = nn.Linear(embed_dim // 2, embed_dim)

        # Movie feature layers
        self.movie_fc = nn.Linear(embed_dim, embed_dim)
        self.category_fc = nn.Linear(embed_dim, embed_dim)

        # Final layers
        self.user_final = nn.Linear(4 * embed_dim, 200)
        self.movie_final = nn.Linear(3 * embed_dim, 200)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(
        self,
        user_id,
        movie_id,
        gender,
        age,
        job_id,
        movie_categories,
        movie_titles,
        poster_img,
    ):
        # User embeddings
        user_embed = self.user_embedding(user_id).view(-1, 1, 32)
        gender_embed = self.gender_embedding(gender).view(-1, 1, 16)
        age_embed = self.age_embedding(age).view(-1, 1, 16)
        job_embed = self.job_embedding(job_id).view(-1, 1, 16)

        # Movie embeddings
        movie_embed = self.movie_embedding(movie_id).view(-1, 1, 32)

        # Process categories - handle each element in batch separately to avoid index errors
        batch_size = movie_categories.size(0)
        category_embeds = []

        for i in range(batch_size):
            # Get categories for this sample
            categories = movie_categories[i]
            # Embed each category
            category_embed = self.category_embedding(categories)
            # Sum the embeddings
            category_embed = torch.sum(category_embed, dim=0, keepdim=True).unsqueeze(0)
            category_embeds.append(category_embed)

        # Stack all category embeddings
        category_embed = torch.cat(category_embeds, dim=0)

        # Process titles using CNN
        title_embed = self.title_embedding(movie_titles)
        title_embed = title_embed.permute(
            0, 2, 1
        )  # Reshape for conv1d [batch, embed_dim, seq_len]

        conv_outputs = []
        for conv in self.title_conv:
            x = F.relu(conv(title_embed))
            x = F.max_pool1d(x, x.size(2))
            conv_outputs.append(x)

        title_features = torch.cat(conv_outputs, dim=1)
        title_features = title_features.view(-1, 1, 32)

        # Process movie poster
        poster_features = self.poster_encoder(poster_img).view(-1, 1, 200)

        # User fully connected layers
        user_fc = F.relu(self.user_fc(user_embed))
        gender_fc = F.relu(self.gender_fc(gender_embed))
        age_fc = F.relu(self.age_fc(age_embed))
        job_fc = F.relu(self.job_fc(job_embed))

        # Movie fully connected layers
        movie_fc = F.relu(self.movie_fc(movie_embed))
        category_fc = F.relu(self.category_fc(category_embed))

        # Combine user features
        user_combined = torch.cat([user_fc, gender_fc, age_fc, job_fc], dim=2)
        user_features = F.tanh(self.user_final(user_combined))
        user_features = user_features.view(-1, 200)

        # Combine movie features
        movie_combined = torch.cat([movie_fc, category_fc, title_features], dim=2)
        movie_features = F.tanh(self.movie_final(movie_combined))
        movie_features = movie_features.view(-1, 200)

        # Add poster features to movie features
        poster_features = poster_features.view(-1, 200)
        movie_features = movie_features + poster_features

        # Final prediction
        prediction = torch.sum(user_features * movie_features, dim=1, keepdim=True)

        return prediction


# 4. Training function
def train_model(train_loader, model, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()

            # Forward pass
            prediction = model(
                batch["user_id"].to(device),
                batch["movie_id"].to(device),
                batch["gender"].to(device),
                batch["age"].to(device),
                batch["job_id"].to(device),
                batch["movie_categories"].to(device),
                batch["movie_titles"].to(device),
                batch["poster_img"].to(device),
            )

            # Compute loss
            loss = criterion(prediction, batch["rating"].view(-1, 1).to(device))

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")


# 5. Generate feature matrices
def generate_movie_features(model, movies, movieid2idx, poster_dir="./poster/"):
    model.eval()
    movie_features = []
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    with torch.no_grad():
        for idx, item in tqdm(
            enumerate(movies.values), desc="Generating movie features"
        ):
            movie_id = item[0]
            categories = torch.tensor(item[2], dtype=torch.long).to(device)
            titles = torch.tensor(item[1], dtype=torch.long).to(device)

            # Load poster image
            poster_path = os.path.join(poster_dir, f"{int(movie_id)}.jpg")
            try:
                poster_img = Image.open(poster_path).convert("RGB")
                poster_img = transform(poster_img)
            except (FileNotFoundError, IOError):
                poster_img = torch.zeros(3, 224, 224)

            poster_img = poster_img.unsqueeze(0).to(device)

            # Get movie embedding
            movie_embed = model.movie_embedding(
                torch.tensor([idx], dtype=torch.long).to(device)
            ).view(-1, 1, 32)

            # Process categories
            category_embed = model.category_embedding(categories)
            category_embed = torch.sum(category_embed, dim=0, keepdim=True).unsqueeze(0)

            # Process titles
            title_embed = model.title_embedding(titles.unsqueeze(0))
            title_embed = title_embed.permute(0, 2, 1)

            conv_outputs = []
            for conv in model.title_conv:
                x = F.relu(conv(title_embed))
                x = F.max_pool1d(x, x.size(2))
                conv_outputs.append(x)

            title_features = torch.cat(conv_outputs, dim=1)
            title_features = title_features.view(-1, 1, 32)

            # Process poster
            poster_features = model.poster_encoder(poster_img).view(-1, 1, 200)

            # Movie fully connected layers
            movie_fc = F.relu(model.movie_fc(movie_embed))
            category_fc = F.relu(model.category_fc(category_embed))

            # Combine movie features
            movie_combined = torch.cat([movie_fc, category_fc, title_features], dim=2)
            movie_features_vec = F.tanh(model.movie_final(movie_combined))
            movie_features_vec = movie_features_vec.view(-1, 200)

            # Add poster features
            poster_features = poster_features.view(-1, 200)
            movie_features_vec = movie_features_vec + poster_features

            movie_features.append(movie_features_vec.cpu().numpy())

    return np.array(movie_features).reshape(-1, 200)


def generate_user_features(model, users):
    model.eval()
    user_features = []

    with torch.no_grad():
        for idx, item in tqdm(enumerate(users.values), desc="Generating user features"):
            user_id = torch.tensor([idx], dtype=torch.long).to(device)
            gender = torch.tensor([item[1]], dtype=torch.long).to(device)
            age = torch.tensor([item[2]], dtype=torch.long).to(device)
            job_id = torch.tensor([item[3]], dtype=torch.long).to(device)

            # User embeddings
            user_embed = model.user_embedding(user_id).view(-1, 1, 32)
            gender_embed = model.gender_embedding(gender).view(-1, 1, 16)
            age_embed = model.age_embedding(age).view(-1, 1, 16)
            job_embed = model.job_embedding(job_id).view(-1, 1, 16)

            # User fully connected layers
            user_fc = F.relu(model.user_fc(user_embed))
            gender_fc = F.relu(model.gender_fc(gender_embed))
            age_fc = F.relu(model.age_fc(age_embed))
            job_fc = F.relu(model.job_fc(job_embed))

            # Combine user features
            user_combined = torch.cat([user_fc, gender_fc, age_fc, job_fc], dim=2)
            user_features_vec = F.tanh(model.user_final(user_combined))
            user_features_vec = user_features_vec.view(-1, 200)

            user_features.append(user_features_vec.cpu().numpy())

    return np.array(user_features).reshape(-1, 200)


# 6. Movie recommendation functions
def recommend_same_type_movie(
    model, movie_id, movie_matrics, movies_orig, movieid2idx, top_k=20
):
    movie_features = torch.tensor(movie_matrics, dtype=torch.float).to(device)
    probs_embeddings = (
        torch.tensor(movie_matrics[movieid2idx[movie_id]], dtype=torch.float)
        .view(1, -1)
        .to(device)
    )

    # Compute similarity
    norm_movie_matrics = torch.sqrt(
        torch.sum(torch.square(movie_features), dim=1, keepdim=True)
    )
    normalized_movie_matrics = movie_features / norm_movie_matrics
    probs_similarity = torch.matmul(probs_embeddings, normalized_movie_matrics.t())

    # Convert to numpy for further processing
    sim = probs_similarity.cpu().numpy()

    print(f"您看的电影是：{movies_orig[movieid2idx[movie_id]]}")
    print("以下是给您的推荐：")

    p = np.squeeze(sim)
    p[np.argsort(p)[:-top_k]] = 0
    p = p / np.sum(p)
    results = set()
    while len(results) != 5:
        c = np.random.choice(len(movie_matrics), 1, p=p)[0]
        results.add(c)

    for val in results:
        print(val)
        print(movies_orig[val])

    return results


def recommend_your_favorite_movie(
    model, user_id, users_matrics, movie_matrics, movies_orig, top_k=10
):
    user_features = (
        torch.tensor(users_matrics[user_id - 1], dtype=torch.float)
        .view(1, -1)
        .to(device)
    )
    movie_features = torch.tensor(movie_matrics, dtype=torch.float).to(device)

    # Compute predictions
    probs_similarity = torch.matmul(user_features, movie_features.t())
    sim = probs_similarity.cpu().numpy()

    print("以下是给您的推荐：")

    p = np.squeeze(sim)
    p[np.argsort(p)[:-top_k]] = 0
    p = p / np.sum(p)
    results = set()
    while len(results) != 5:
        c = np.random.choice(len(movie_matrics), 1, p=p)[0]
        results.add(c)

    for val in results:
        print(val)
        print(movies_orig[val])

    return results


def recommend_other_favorite_movie(
    model,
    movie_id,
    users_matrics,
    movie_matrics,
    movies_orig,
    users_orig,
    movieid2idx,
    top_k=20,
):
    movie_features = (
        torch.tensor(movie_matrics[movieid2idx[movie_id]], dtype=torch.float)
        .view(1, -1)
        .to(device)
    )
    user_features = torch.tensor(users_matrics, dtype=torch.float).to(device)

    # Find users who like this movie
    probs_user_favorite_similarity = torch.matmul(movie_features, user_features.t())
    favorite_user_ids = torch.argsort(probs_user_favorite_similarity.cpu().squeeze())[
        -top_k:
    ].numpy()

    print(f"您看的电影是：{movies_orig[movieid2idx[movie_id]]}")
    print(f"喜欢看这个电影的人是：{users_orig[favorite_user_ids]}")

    # Get favorite movies for those users
    selected_users = torch.tensor(
        users_matrics[favorite_user_ids], dtype=torch.float
    ).to(device)
    selected_movies = torch.tensor(movie_matrics, dtype=torch.float).to(device)

    # Find what movies these users like
    probs_similarity = torch.matmul(selected_users, selected_movies.t())
    liked_movies = torch.argmax(probs_similarity, dim=1).cpu().numpy()

    print("喜欢看这个电影的人还喜欢看：")

    if len(set(liked_movies)) < 5:
        results = set(liked_movies)
    else:
        results = set()
        while len(results) != 5:
            c = liked_movies[random.randrange(top_k)]
            results.add(c)

    for val in results:
        print(val)
        print(movies_orig[val])

    return results


# Main function
def main():
    # Load data
    (
        title_count,
        title_set,
        genres2int,
        features,
        targets_values,
        ratings,
        users,
        movies,
        data,
        movies_orig,
        users_orig,
        movieid2idx,
    ) = load_data()

    # Create dataset and dataloader
    dataset = MovieLensDataset(features, targets_values, movies, title_count)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    # Initialize model
    n_users = int(np.max(users.values[:, 0]))  # 0-indexed
    n_movies = int(np.max(movies.values[:, 0]))  # 0-indexed
    n_genders = 2  # 0: Female, 1: Male
    n_ages = len(set(users["Age"]))
    n_jobs = len(set(users["JobID"])) + 1  # +1 for potential unknown jobs
    n_categories = (
        max(
            max(genres2int.values()),
            np.max([max(x) if len(x) > 0 else 0 for x in movies["Genres"]]),
        )
        + 1
    )
    n_titles = len(title_set)

    print(
        f"n_users: {n_users}, n_movies: {n_movies}, n_categories: {n_categories}, n_titles: {n_titles}"
    )

    model = MovieRecommendationModel(
        n_users, n_movies, n_genders, n_ages, n_jobs, n_categories, n_titles
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    train_model(train_loader, model, criterion, optimizer, epochs=5)

    # Save model
    torch.save(model.state_dict(), "movie_recommendation_model.pth")

    # Generate features
    movie_matrics = generate_movie_features(model, movies, movieid2idx)
    users_matrics = generate_user_features(model, users)

    # Save features
    pickle.dump(movie_matrics, open("movie_matrics.p", "wb"))
    pickle.dump(users_matrics, open("users_matrics.p", "wb"))

    # Example recommendations
    print("\n--- Recommendations for similar movies ---")
    recommend_same_type_movie(model, 1401, movie_matrics, movies_orig, movieid2idx)

    print("\n--- Recommendations for user ---")
    recommend_your_favorite_movie(model, 234, users_matrics, movie_matrics, movies_orig)

    print("\n--- Recommendations based on other users ---")
    recommend_other_favorite_movie(
        model, 1401, users_matrics, movie_matrics, movies_orig, users_orig, movieid2idx
    )


if __name__ == "__main__":
    main()
