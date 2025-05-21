<template>
  <div class="container">
    <h1 class="title">🎬 多模态电影推荐系统</h1>

    <div class="card">
      <h2>请输入用户信息</h2>
      <form @submit.prevent="getRecommendations">
        <label>
          用户id：
          <input type="text" v-model="userName" required />
        </label>
        <button type="submit">获取推荐</button>
      </form>
    </div>

    <!-- 弹窗组件 -->
    <div class="modal" v-if="showModal" @click.self="closeModal">
      <div class="modal-content">
        <div class="modal-header">
          <h2>🎉 推荐结果</h2>
          <button class="close-button" @click="closeModal">&times;</button>
        </div>
        <div class="modal-body">
          <div class="modal-grid">
            <!-- 左侧用户信息 -->
            <div class="user-info">
              <h3>用户信息</h3>
              <div class="info-item">
                <span class="label">user_id：</span>
                <span class="value">{{ userInfo.id }}</span>
              </div>
              <div class="info-item">
                <span class="label">年龄：</span>
                <span class="value">{{ userInfo.age }}</span>
              </div>
              <div class="info-item">
                <span class="label">性别：</span>
                <span class="value">{{ userInfo.gender }}</span>
              </div>
              <div class="info-item">
                <span class="label">job_id：</span>
                <span class="value">{{ userInfo.job_id }}</span>
              </div>
            </div>
            
            <!-- 右侧电影推荐 -->
            <div class="movie-recommendations">
              <h3>为您推荐</h3>
              <div class="movie-grid">
                <div v-for="(movie, index) in recommendations" :key="index" class="movie-card">
                  <img :src="movie.poster" :alt="movie.title" class="movie-poster">
                  <div class="movie-title">{{ movie.title }}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <footer>
      <p>开发团队：DM_CCL</p>
      <p>开发人员：张凯博、祁瑜、沈超远、刘惟楚、李焱杰</p>
    </footer>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const userName = ref('')
const userInfo = ref({
  id: '',
  age: '',
  gender: '',
  job_id: ''
})
const recommendations = ref([])
const showModal = ref(false)

// 动态导入图片的函数
function getImageUrl(number) {
  return new URL(`./assets/images/poster/${number}.jpg`, import.meta.url).href
}

async function getRecommendations() {
  try {
    // 调用后端 API 获取推荐
    const response = await fetch(`http://localhost:5000/get_recommendations?user_id=${userName.value}`);
    const data = await response.json();
    
    // 打印返回的数据
    console.log('后端返回的数据:', data);
    
    // 更新用户信息
    userInfo.value = {
      id: data.user_info.user_id,
      age: data.user_info.age,
      gender: data.user_info.gender === 'M' ? '男' : '女',
      job_id: data.user_info.job_id
    }
    
    // 更新推荐电影列表
    recommendations.value = data.recommended_movies.map(movie => ({
      title: movie.title,
      poster: getImageUrl(movie.movie_id) // 使用 movie_id 来确定海报图片
    }));
    
    showModal.value = true;
  } catch (error) {
    console.error('获取推荐失败:', error);
    alert('获取推荐失败，请稍后重试');
  }
}

function closeModal() {
  showModal.value = false
}
</script>

<style scoped>
.container {
  max-width: 1200px;
  margin: auto;
  padding: 3rem;
  font-family: "Helvetica Neue", sans-serif;
  font-size: 1.2rem;
  box-sizing: border-box;
}

.title {
  text-align: center;
  font-size: 3rem;
  margin-bottom: 3rem;
}

.card {
  background-color: #f9f9f9;
  padding: 2.5rem;
  margin-bottom: 2rem;
  border-radius: 16px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  color: #333;
  box-sizing: border-box;
}

.card h2 {
  color: #1a1a1a;
  margin-bottom: 1.5rem;
  font-size: 1.8rem;
}

label {
  display: block;
  margin-bottom: 1.5rem;
  font-size: 1.3rem;
  color: #333;
  width: 100%;
}

input, select {
  width: 100%;
  padding: 1rem;
  margin-top: 0.5rem;
  border: 1px solid #ccc;
  border-radius: 8px;
  font-size: 1.2rem;
  color: #333;
  background-color: white;
  box-sizing: border-box;
}

button {
  padding: 1rem 2.5rem;
  background-color: #3b82f6;
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 1.3rem;
  margin-top: 1rem;
}

button:hover {
  background-color: #2563eb;
}

footer {
  text-align: center;
  margin-top: 2rem;
  color: #666;
  font-size: 1.2rem;
}

footer p {
  margin: 0.5rem 0;
}

.card ul {
  list-style: none;
  padding: 0;
}

.card li {
  padding: 1rem;
  margin-bottom: 0.8rem;
  background-color: white;
  border-radius: 8px;
  color: #333;
  font-size: 1.2rem;
}

/* 弹窗样式 */
.modal {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.modal-content {
  background-color: white;
  padding: 2rem;
  border-radius: 16px;
  width: 95%;
  max-width: 1200px;
  max-height: 90vh;
  overflow-y: auto;
  position: relative;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}

.modal-header h2 {
  margin: 0;
  font-size: 1.8rem;
  color: #1a1a1a;
}

.close-button {
  background: none;
  border: none;
  font-size: 2rem;
  color: #666;
  cursor: pointer;
  padding: 0;
  margin: 0;
  line-height: 1;
}

.close-button:hover {
  color: #333;
}

.modal-body {
  font-size: 1.2rem;
}

.modal-body ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

.modal-body li {
  padding: 1rem;
  margin-bottom: 0.8rem;
  background-color: #f9f9f9;
  border-radius: 8px;
  color: #333;
}

.modal-grid {
  display: grid;
  grid-template-columns: 1fr 2fr;
  gap: 2rem;
  padding: 1rem 0;
}

.user-info {
  background-color: #2c3e50;
  padding: 1.5rem;
  border-radius: 12px;
  color: white;
}

.user-info h3 {
  margin-top: 0;
  margin-bottom: 1.5rem;
  font-size: 1.5rem;
  color: white;
}

.info-item {
  margin-bottom: 1rem;
  font-size: 1.2rem;
  color: white;
}

.info-item .label {
  font-weight: bold;
  color: #a0aec0;
  margin-right: 0.5rem;
}

.info-item .value {
  color: white;
}

.movie-recommendations {
  background-color: #2c3e50;
  padding: 1.5rem;
  border-radius: 12px;
  color: white;
}

.movie-recommendations h3 {
  margin-top: 0;
  margin-bottom: 1.5rem;
  font-size: 1.5rem;
  color: white;
}

.movie-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 1.5rem;
}

.movie-card {
  background-color: #34495e;
  border-radius: 12px;
  overflow: hidden;
  transition: transform 0.2s;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}

.movie-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.movie-poster {
  width: 100%;
  height: 300px;
  object-fit: cover;
}

.movie-title {
  padding: 1rem;
  font-size: 1.1rem;
  color: white;
  text-align: center;
}

@media (max-width: 768px) {
  .modal-grid {
    grid-template-columns: 1fr;
  }
  
  .modal-content {
    width: 95%;
    max-height: 95vh;
  }
}
</style>
