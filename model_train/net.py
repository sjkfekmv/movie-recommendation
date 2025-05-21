import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义推荐系统模型（保持原模型定义不变）
class MovieRecommendationModel(nn.Module):
    def __init__(self, n_users, n_movies, n_genders, n_ages, n_jobs, n_categories, n_titles, embed_dim=32, title_dim=15):
        super(MovieRecommendationModel, self).__init__()
        # User embeddings
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.gender_embedding = nn.Embedding(n_genders, embed_dim // 2)
        self.age_embedding = nn.Embedding(n_ages, embed_dim // 2)
        self.job_embedding = nn.Embedding(n_jobs, embed_dim // 2)
        self.embed_dim = embed_dim
        # Movie embeddings
        self.movie_embedding = nn.Embedding(n_movies, embed_dim)
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
            nn.Linear(64, embed_dim)
        )
        
        # CNN for titles
        self.title_conv = nn.ModuleList([
            nn.Conv1d(embed_dim, 8, kernel_size=k) for k in [2, 3, 4, 5]
        ])
        # Calculate title feature dimension (8 channels per conv, 4 convs)
        self.title_feature_dim = 8 * len(self.title_conv)  # 8 * 4 = 32
        self.title_fc = nn.Linear(self.title_feature_dim, embed_dim)  # Project to embed_dim
        
        # Cross-modal attention for movie and poster features
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4)
        
        # Fusion layers
        self.fusion_fc1 = nn.Linear(embed_dim * 2, embed_dim)
        self.fusion_fc2 = nn.Linear(embed_dim, embed_dim)
        
        # User feature layers
        self.user_fc = nn.Linear(embed_dim, embed_dim)
        self.gender_fc = nn.Linear(embed_dim // 2, embed_dim)
        self.age_fc = nn.Linear(embed_dim // 2, embed_dim)
        self.job_fc = nn.Linear(embed_dim // 2, embed_dim)
        
        # Movie feature layers
        self.movie_fc = nn.Linear(embed_dim, embed_dim)
        self.category_fc = nn.Linear(embed_dim, embed_dim)
        
        # Final layers
        self.user_final = nn.Linear(4 * embed_dim, embed_dim)
        self.movie_final = nn.Linear(embed_dim, embed_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, user_id, movie_id, gender, age, job_id, movie_categories, movie_titles, poster_img):
        # User embeddings
        user_embed = self.user_embedding(user_id)  # [batch, embed_dim]
        gender_embed = self.gender_embedding(gender)  # [batch, embed_dim//2]
        age_embed = self.age_embedding(age)  # [batch, embed_dim//2]
        job_embed = self.job_embedding(job_id)  # [batch, embed_dim//2]
        
        # Movie embeddings
        movie_embed = self.movie_embedding(movie_id)  # [batch, embed_dim]
        
        # Process categories
        batch_size = movie_categories.size(0)
        category_embeds = []
        for i in range(batch_size):
            categories = movie_categories[i]
            category_embed = self.category_embedding(categories)  # [num_categories, embed_dim]
            category_embed = torch.sum(category_embed, dim=0, keepdim=True)  # [1, embed_dim]
            category_embeds.append(category_embed)
        category_embed = torch.cat(category_embeds, dim=0)  # [batch, embed_dim]
        
        # Process titles using CNN
        title_embed = self.title_embedding(movie_titles)  # [batch, seq_len, embed_dim]
        title_embed = title_embed.permute(0, 2, 1)  # [batch, embed_dim, seq_len]
        conv_outputs = []
        for conv in self.title_conv:
            x = F.relu(conv(title_embed))  # [batch, 8, seq_len-k+1]
            x = F.max_pool1d(x, x.size(2))  # [batch, 8, 1]
            conv_outputs.append(x)
        title_features = torch.cat(conv_outputs, dim=1)  # [batch, title_feature_dim]
        title_features = title_features.view(-1, self.title_feature_dim)  # [batch, 32]
        title_features = F.relu(self.title_fc(title_features))  # [batch, embed_dim]
        
        # Process movie poster
        poster_features = self.poster_encoder(poster_img)  # [batch, embed_dim]
        
        # Combine movie features (before attention)
        movie_combined = movie_embed + category_embed + title_features  # [batch, embed_dim]
        
        # Cross-modal attention
        movie_combined = movie_combined.unsqueeze(0)  # [1, batch, embed_dim]
        poster_features = poster_features.unsqueeze(0)  # [1, batch, embed_dim]
        attn_output, _ = self.cross_attention(movie_combined, poster_features, poster_features)
        attn_output = attn_output.squeeze(0)  # [batch, embed_dim]
        
        # Feature fusion
        fusion_input = torch.cat([attn_output, poster_features.squeeze(0)], dim=1)  # [batch, embed_dim*2]
        fusion_output = F.relu(self.fusion_fc1(fusion_input))  # [batch, embed_dim]
        fusion_output = F.relu(self.fusion_fc2(fusion_output))  # [batch, embed_dim]
        fusion_output = self.dropout(fusion_output)
        
        # User fully connected layers
        user_fc = F.relu(self.user_fc(user_embed))  # [batch, embed_dim]
        gender_fc = F.relu(self.gender_fc(gender_embed))  # [batch, embed_dim]
        age_fc = F.relu(self.age_fc(age_embed))  # [batch, embed_dim]
        job_fc = F.relu(self.job_fc(job_embed))  # [batch, embed_dim]
        
        # Combine user features
        user_combined = torch.cat([user_fc, gender_fc, age_fc, job_fc], dim=1)  # [batch, embed_dim*4]
        user_features = F.tanh(self.user_final(user_combined))  # [batch, embed_dim]
        user_features = self.dropout(user_features)
        
        # Final movie features
        movie_features = F.tanh(self.movie_final(fusion_output))  # [batch, embed_dim]
        
        # Final prediction
        prediction = torch.sum(user_features * movie_features, dim=1, keepdim=True)
        
        return prediction

# 打印网络结构
def print_network_structure(n_users, n_movies, n_genders, n_ages, n_jobs, n_categories, n_titles, embed_dim=32):
    # 实例化模型
    model = MovieRecommendationModel(
        n_users=n_users,
        n_movies=n_movies,
        n_genders=n_genders,
        n_ages=n_ages,
        n_jobs=n_jobs,
        n_categories=n_categories,
        n_titles=n_titles,
        embed_dim=embed_dim
    )
    
    # 打印模型结构
    print("=== 推荐系统模型结构 ===")
    print(model)
    
    # 打印参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数数量: {total_params:,}")
    
    # 打印每一层的参数详情
    print("\n=== 参数详情 ===")
    for name, param in model.named_parameters():
        print(f"层: {name}, 形状: {param.shape}, 参数数量: {param.numel():,}")

# 主函数
if __name__ == "__main__":
    # 基于 MovieLens-1M 数据集的参数
    n_users = 6040      # 用户数量
    n_movies = 3952     # 电影数量
    n_genders = 2       # 性别（男/女）
    n_ages = 7          # 年龄段（1, 18, 25, 35, 45, 50, 56）
    n_jobs = 21         # 职业类别（0-20）
    n_categories = 18   # 电影类别数量（Action, Comedy, Drama 等）
    n_titles = 5000     # 电影标题词汇表大小（假设，需根据实际预处理调整）
    embed_dim = 32      # 嵌入维度（默认值）

    # 打印网络结构
    print_network_structure(n_users, n_movies, n_genders, n_ages, n_jobs, n_categories, n_titles, embed_dim)