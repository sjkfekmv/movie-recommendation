import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from skimage.feature import graycomatrix, graycoprops
from collections import Counter
import cv2
from sklearn.preprocessing import StandardScaler

# 设置随机种子，确保结果可复现
np.random.seed(42)
torch.manual_seed(42)

# 定义数据路径
MOVIELENS_PATH = "ml-1m/"  # MovieLens数据集路径
POSTERS_PATH = "poster/"  # 电影海报路径

class PosterFeatureExtractor:
    def __init__(self):
        # 加载预训练的ResNet-18模型
        self.model = models.resnet18(pretrained=True)
        
        # 只保留浅层特征（前几层卷积层）
        self.shallow_model = nn.Sequential(*list(self.model.children())[:6])
        self.shallow_model.eval()
        
        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def extract_shallow_features(self, img_path):
        """提取浅层特征"""
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0)
            
            with torch.no_grad():
                features = self.shallow_model(img_tensor)
            
            # 转换为特征向量
            features = features.mean([2, 3]).squeeze(0).numpy()
            return features, img
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None, None
    
    def extract_color_features(self, img):
        """提取颜色分布特征"""
        img_np = np.array(img)
        # 转换为HSV颜色空间
        hsv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        
        # 提取每个通道的直方图
        h_hist = np.histogram(hsv_img[:,:,0], bins=16, range=(0, 180))[0]
        s_hist = np.histogram(hsv_img[:,:,1], bins=16, range=(0, 256))[0]
        v_hist = np.histogram(hsv_img[:,:,2], bins=16, range=(0, 256))[0]
        
        # 归一化
        h_hist = h_hist / h_hist.sum() if h_hist.sum() > 0 else h_hist
        s_hist = s_hist / s_hist.sum() if s_hist.sum() > 0 else s_hist
        v_hist = v_hist / v_hist.sum() if v_hist.sum() > 0 else v_hist
        
        # 计算颜色统计特征
        avg_hue = hsv_img[:,:,0].mean()
        avg_saturation = hsv_img[:,:,1].mean()
        avg_value = hsv_img[:,:,2].mean()
        
        # 检测是否是暖色调主导
        # 暖色调在HSV中大致为H: 0-60或300-360，对应于红、橙、黄色
        warm_mask = ((hsv_img[:,:,0] <= 60) | (hsv_img[:,:,0] >= 300)) & (hsv_img[:,:,1] > 70)
        cool_mask = ((hsv_img[:,:,0] > 60) & (hsv_img[:,:,0] < 300)) & (hsv_img[:,:,1] > 70)
        warm_ratio = np.sum(warm_mask) / (np.sum(warm_mask) + np.sum(cool_mask) + 1e-10)
        
        return {
            'h_hist': h_hist,
            's_hist': s_hist,
            'v_hist': v_hist,
            'avg_hue': avg_hue,
            'avg_saturation': avg_saturation,
            'avg_value': avg_value,
            'warm_ratio': warm_ratio
        }
    
    def extract_texture_features(self, img):
        """提取纹理特征"""
        img_np = np.array(img)
        # 转换为灰度图
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        # 调整大小，加快计算速度
        gray = cv2.resize(gray, (100, 100))
        
        # 计算灰度共生矩阵和特征
        distances = [1, 3, 5]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
        
        # 提取共生矩阵特征
        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        
        # 使用Canny边缘检测算法
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # 使用Sobel算子计算局部变化
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        complexity = np.mean(sobel_magnitude)
        
        return {
            'contrast': contrast,
            'dissimilarity': dissimilarity,
            'homogeneity': homogeneity,
            'energy': energy,
            'correlation': correlation,
            'edge_density': edge_density,
            'complexity': complexity
        }
    
    def extract_composition_features(self, img):
        """提取构图特征"""
        img_np = np.array(img)
        height, width = img_np.shape[:2]
        
        # 图像分为9个区域
        h_segments = 3
        w_segments = 3
        h_step = height // h_segments
        w_step = width // w_segments
        
        segment_intensities = []
        for i in range(h_segments):
            for j in range(w_segments):
                h_start, h_end = i * h_step, (i + 1) * h_step
                w_start, w_end = j * w_step, (j + 1) * w_step
                segment = img_np[h_start:h_end, w_start:w_end]
                segment_intensities.append(segment.mean())
        
        # 中心-边缘比
        center_segment = img_np[h_step:2*h_step, w_step:2*w_step]
        center_intensity = center_segment.mean()
        edge_intensity = (img_np.mean() * img_np.size - center_segment.mean() * center_segment.size) / (img_np.size - center_segment.size)
        center_edge_ratio = center_intensity / (edge_intensity + 1e-10)
        
        # 使用Hough变换检测线条
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        num_lines = 0 if lines is None else len(lines)
        
        # 计算边缘密度
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # 计算简洁度指标（边缘密度的倒数）
        simplicity = 1.0 / (edge_density + 1e-10)
        
        return {
            'segment_intensities': segment_intensities,
            'center_edge_ratio': center_edge_ratio,
            'num_lines': num_lines,
            'simplicity': simplicity
        }

def load_movielens_data():
    """加载MovieLens数据集"""
    # 加载电影数据
    movies_df = pd.read_csv(os.path.join(MOVIELENS_PATH, 'movies.dat'), 
                           sep='::', 
                           names=['MovieID', 'Title', 'Genres'],
                           encoding='latin-1',
                           engine='python')
    
    # 加载评分数据
    ratings_df = pd.read_csv(os.path.join(MOVIELENS_PATH, 'ratings.dat'),
                            sep='::',
                            names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
                            encoding='latin-1',
                            engine='python')
    
    # 计算每部电影的平均评分
    movie_ratings = ratings_df.groupby('MovieID')['Rating'].agg(['mean', 'count']).reset_index()
    movie_ratings.columns = ['MovieID', 'AvgRating', 'RatingCount']
    
    # 合并电影信息和评分
    movies_with_ratings = pd.merge(movies_df, movie_ratings, on='MovieID')
    
    # 处理类型信息
    genres_set = set()
    for genres in movies_df['Genres'].str.split('|'):
        genres_set.update(genres)
    
    # 为每个电影创建类型标签
    for genre in genres_set:
        movies_with_ratings[genre] = movies_with_ratings['Genres'].apply(lambda x: 1 if genre in x.split('|') else 0)
    
    return movies_with_ratings

def process_all_posters(movies_df):
    """处理所有电影海报"""
    extractor = PosterFeatureExtractor()
    all_features = {}
    
    # 检查POSTERS_PATH是否存在
    if not os.path.exists(POSTERS_PATH):
        print(f"Error: Posters directory '{POSTERS_PATH}' not found.")
        print(f"Creating directory '{POSTERS_PATH}'...")
        os.makedirs(POSTERS_PATH)
        print(f"Please place your movie poster images in the '{POSTERS_PATH}' directory.")
        return all_features
    
    # 检查是否有图片文件
    poster_files = [f for f in os.listdir(POSTERS_PATH) if f.endswith('.jpg')]
    if len(poster_files) == 0:
        print(f"Warning: No JPG files found in '{POSTERS_PATH}' directory.")
        return all_features
    
    print(f"Found {len(poster_files)} poster images in '{POSTERS_PATH}'")
    
    processed_count = 0
    for _, row in tqdm(movies_df.iterrows(), total=len(movies_df), desc="Processing posters"):
        movie_id = row['MovieID']
        poster_path = os.path.join(POSTERS_PATH, f"{movie_id}.jpg")
        
        if os.path.exists(poster_path):
            try:
                # 提取特征
                resnet_features, img = extractor.extract_shallow_features(poster_path)
                
                if resnet_features is not None and img is not None:
                    # 检查ResNet特征的形状
                    if not isinstance(resnet_features, np.ndarray):
                        print(f"Warning: Invalid ResNet features for movie ID {movie_id}. Skipping.")
                        continue
                    
                    # 提取颜色特征
                    color_features = extractor.extract_color_features(img)
                    
                    # 提取纹理特征
                    texture_features = extractor.extract_texture_features(img)
                    
                    # 提取构图特征
                    try:
                        composition_features = extractor.extract_composition_features(img)
                    except NameError:
                        # 如果edge_density未定义，先计算它
                        edge_density = texture_features['edge_density']
                        composition_features = {
                            'segment_intensities': [],
                            'center_edge_ratio': 1.0,
                            'num_lines': 0,
                            'simplicity': 1.0 / (edge_density + 1e-10)
                        }
                    
                    all_features[movie_id] = {
                        'resnet_features': resnet_features,
                        'color_features': color_features,
                        'texture_features': texture_features,
                        'composition_features': composition_features
                    }
                    processed_count += 1
            except Exception as e:
                print(f"Error processing poster for movie ID {movie_id}: {e}")
    
    print(f"Successfully processed {processed_count} out of {len(poster_files)} poster images")
    return all_features

def visualize_features(movies_df, all_features):
    """可视化特征分析结果"""
    # 检查是否有足够的特征数据
    if len(all_features) == 0:
        print("Error: No features were extracted from movie posters. Please check your poster images.")
        return
    
    # 准备数据
    movie_ids = []
    resnet_features_list = []
    color_features_dict = {'warm_ratio': [], 'avg_saturation': [], 'avg_value': []}
    texture_features_dict = {'contrast': [], 'homogeneity': [], 'edge_density': [], 'complexity': []}
    composition_features_dict = {'center_edge_ratio': [], 'simplicity': []}
    genres = []
    ratings = []
    
    for movie_id, features in all_features.items():
        try:
            # 将movie_id转换为整数，因为DataFrame中可能是整数类型
            movie_id_int = int(movie_id)
            movie_row = movies_df[movies_df['MovieID'] == movie_id_int]
            if movie_row.empty:
                print(f"Warning: Movie ID {movie_id} not found in dataset. Skipping.")
                continue
            movie_row = movie_row.iloc[0]
            
            # 检查ResNet特征是否有效
            if features['resnet_features'] is None or not isinstance(features['resnet_features'], np.ndarray):
                print(f"Warning: Invalid ResNet features for movie ID {movie_id}. Skipping.")
                continue
                
            movie_ids.append(movie_id)
            resnet_features_list.append(features['resnet_features'])
            
            # 颜色特征
            for key in color_features_dict.keys():
                if key in features['color_features']:
                    color_features_dict[key].append(features['color_features'][key])
                else:
                    print(f"Warning: Color feature '{key}' missing for movie ID {movie_id}")
                    color_features_dict[key].append(0.0)  # 使用默认值
            
            # 纹理特征
            for key in texture_features_dict.keys():
                if key in features['texture_features']:
                    texture_features_dict[key].append(features['texture_features'][key])
                else:
                    print(f"Warning: Texture feature '{key}' missing for movie ID {movie_id}")
                    texture_features_dict[key].append(0.0)  # 使用默认值
            
            # 构图特征
            for key in composition_features_dict.keys():
                if key in features['composition_features']:
                    composition_features_dict[key].append(features['composition_features'][key])
                else:
                    print(f"Warning: Composition feature '{key}' missing for movie ID {movie_id}")
                    composition_features_dict[key].append(0.0)  # 使用默认值
            
            # 类型和评分
            genres.append(movie_row['Genres'])
            ratings.append(movie_row['AvgRating'])
        except Exception as e:
            print(f"Error processing movie ID {movie_id}: {e}")
    
    # 检查是否有足够的数据点进行可视化
    if len(resnet_features_list) == 0:
        print("Error: No valid features could be extracted. Please check your data.")
        return
    
    print(f"Collected features for {len(resnet_features_list)} movies")
    
    # 转换为数组
    resnet_features_array = np.array(resnet_features_list)
    ratings_array = np.array(ratings)
    
    # 检查数据形状
    print(f"Shape of ResNet features array: {resnet_features_array.shape}")
    if len(resnet_features_array.shape) == 1 or resnet_features_array.shape[0] == 0:
        print("Error: ResNet features array is empty or has incorrect shape.")
        return
        
    # 检查所有ResNet特征向量的长度是否一致
    feature_lengths = [len(feat) for feat in resnet_features_list]
    if len(set(feature_lengths)) > 1:
        print(f"Warning: Inconsistent feature vector lengths detected: {set(feature_lengths)}")
        # 找出最常见的长度
        most_common_length = max(set(feature_lengths), key=feature_lengths.count)
        print(f"Using the most common length: {most_common_length}")
        # 过滤掉长度不一致的特征
        filtered_indices = [i for i, feat in enumerate(resnet_features_list) if len(feat) == most_common_length]
        resnet_features_list = [resnet_features_list[i] for i in filtered_indices]
        ratings = [ratings[i] for i in filtered_indices]
        genres = [genres[i] for i in filtered_indices]
        for key in color_features_dict:
            color_features_dict[key] = [color_features_dict[key][i] for i in filtered_indices]
        for key in texture_features_dict:
            texture_features_dict[key] = [texture_features_dict[key][i] for i in filtered_indices]
        for key in composition_features_dict:
            composition_features_dict[key] = [composition_features_dict[key][i] for i in filtered_indices]
            
        # 重新创建数组
        resnet_features_array = np.array(resnet_features_list)
        ratings_array = np.array(ratings)
        print(f"After filtering: {len(resnet_features_list)} movies with consistent feature length")
    
    # 1. 可视化ResNet特征的降维结果
    # 使用PCA降维
    try:
        pca = PCA(n_components=2)
        resnet_pca = pca.fit_transform(resnet_features_array)
        print("PCA completed successfully")
    except Exception as e:
        print(f"Error during PCA: {e}")
        print("Trying alternative approach...")
        
        try:
            # 标准化特征
            scaler = StandardScaler()
            resnet_features_scaled = scaler.fit_transform(resnet_features_array)
            pca = PCA(n_components=2)
            resnet_pca = pca.fit_transform(resnet_features_scaled)
            print("PCA with scaling completed successfully")
        except Exception as e2:
            print(f"Error during scaled PCA: {e2}")
            print("PCA visualization will be skipped")
            resnet_pca = None
    
    plt.figure(figsize=(12, 10))
    
    # 将电影分为高评分和低评分
    high_rating_mask = ratings_array >= 4.0
    low_rating_mask = ratings_array < 3.0
    
    plt.scatter(resnet_pca[high_rating_mask, 0], resnet_pca[high_rating_mask, 1], 
                c='red', label='High Rating (≥4.0)', alpha=0.7)
    plt.scatter(resnet_pca[low_rating_mask, 0], resnet_pca[low_rating_mask, 1], 
                c='blue', label='Low Rating (<3.0)', alpha=0.7)
    
    plt.title('PCA of ResNet Features by Movie Rating')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.savefig('resnet_pca_by_rating.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 不同类型电影的颜色特征对比
    action_mask = [1 if 'Action' in genre else 0 for genre in genres]
    comedy_mask = [1 if 'Comedy' in genre else 0 for genre in genres]
    
    action_mask = np.array(action_mask).astype(bool)
    comedy_mask = np.array(comedy_mask).astype(bool)
    
    # 颜色特征比较 - 暖色调比例
    plt.figure(figsize=(10, 6))
    action_warm = np.array(color_features_dict['warm_ratio'])[action_mask]
    comedy_warm = np.array(color_features_dict['warm_ratio'])[comedy_mask]
    
    plt.hist(action_warm, bins=20, alpha=0.7, label='Action Movies', color='darkred')
    plt.hist(comedy_warm, bins=20, alpha=0.7, label='Comedy Movies', color='gold')
    plt.title('Distribution of Warm Colors Ratio in Action vs Comedy Movies')
    plt.xlabel('Warm Colors Ratio')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('warm_colors_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 纹理特征比较
    plt.figure(figsize=(14, 8))
    
    # 转换为数组以便计算
    texture_arrays = {k: np.array(v) for k, v in texture_features_dict.items()}
    
    features_to_plot = ['contrast', 'homogeneity', 'edge_density', 'complexity']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(features_to_plot):
        ax = axes[i]
        action_values = texture_arrays[feature][action_mask]
        comedy_values = texture_arrays[feature][comedy_mask]
        
        ax.hist(action_values, bins=20, alpha=0.7, label='Action', color='darkred')
        ax.hist(comedy_values, bins=20, alpha=0.7, label='Comedy', color='gold')
        ax.set_title(f'{feature.capitalize()} Distribution: Action vs Comedy')
        ax.set_xlabel(feature.capitalize())
        ax.set_ylabel('Count')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('texture_features_comparison.png', dpi=300)
    plt.close()
    
    # 4. 评分与视觉特征的关系
    plt.figure(figsize=(12, 10))
    
    # 评分与暖色调比例的散点图
    plt.scatter(np.array(color_features_dict['warm_ratio']), ratings_array, 
                alpha=0.5, c=np.array(color_features_dict['avg_saturation']))
    plt.colorbar(label='Average Saturation')
    plt.title('Relationship Between Warm Colors Ratio and Movie Ratings')
    plt.xlabel('Warm Colors Ratio')
    plt.ylabel('Average Rating')
    plt.savefig('rating_vs_warm_colors.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 评分与构图简洁性的关系
    plt.figure(figsize=(12, 10))
    plt.scatter(np.array(composition_features_dict['simplicity']), ratings_array, 
                alpha=0.5, c=np.array(color_features_dict['avg_value']))
    plt.colorbar(label='Average Brightness')
    plt.title('Relationship Between Composition Simplicity and Movie Ratings')
    plt.xlabel('Simplicity Score')
    plt.ylabel('Average Rating')
    plt.savefig('rating_vs_simplicity.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 加载MovieLens数据
    print("Loading MovieLens data...")
    try:
        movies_df = load_movielens_data()
        print(f"Loaded {len(movies_df)} movies")
    except Exception as e:
        print(f"Error loading MovieLens data: {e}")
        print("Please make sure the 'ml-1m' directory exists with the proper files.")
        return
    
    # 处理所有海报
    print("Processing movie posters...")
    try:
        all_features = process_all_posters(movies_df)
        if len(all_features) == 0:
            print("No features were extracted. Please check your poster images and directory structure.")
            print(f"Expected directory structure: '{POSTERS_PATH}/[movie_id].jpg'")
            return
        print(f"Processed {len(all_features)} movie posters")
    except Exception as e:
        print(f"Error processing movie posters: {e}")
        return
    
    # 可视化分析结果
    print("Visualizing analysis results...")
    try:
        visualize_features(movies_df, all_features)
        print("Visualization complete. Results saved as PNG files.")
    except Exception as e:
        print(f"Error during visualization: {e}")
    
    # 保存特征为numpy文件供后续使用
    print("Saving extracted features...")
    try:
        # 使用pickle保存，因为numpy不能直接保存包含字典的复杂对象
        import pickle
        with open('movie_poster_features.pkl', 'wb') as f:
            pickle.dump(all_features, f)
        print("Features saved to movie_poster_features.pkl")
    except Exception as e:
        print(f"Error saving features: {e}")
        
    print("\nData preprocessing complete!")
    print("Tips for next steps:")
    print("- Check the generated visualization files to see the visual differences between genres")
    print("- For model building, load the features using 'with open('movie_poster_features.pkl', 'rb') as f: features = pickle.load(f)'")
    print("- Consider integrating these visual features with collaborative filtering for multimodal recommendations")

if __name__ == "__main__":
    main()