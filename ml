import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import numpy as np

# -------------------------------
# 1. 從 Google Drive 讀取 CSV
# -------------------------------
url = "https://drive.google.com/uc?id=1w-4vVxwSbhvIdU-ayfxa8_BypFP7VXKo&export=download"
df = pd.read_csv(url)

print("="*50)
print("資料前五筆")
print("="*50)
print(df.head())

# -------------------------------
# 2. 前處理
# -------------------------------
categorical_cols = ['gender','diet_quality','parental_education_level',
                    'internet_quality','extracurricular_participation','part_time_job']

for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# 特徵與目標
X = df.drop(columns=['student_id','exam_score'])
y = df['exam_score']

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 3. 監督式學習：線性迴歸
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# 相容版本 RMSE 計算
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("\n" + "="*50)
print("監督式學習結果 (Linear Regression)")
print("="*50)
print(f"RMSE: {rmse:.2f}")

# -------------------------------
# 4. 非監督式學習：KMeans 分群
# -------------------------------
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df['cluster'] = clusters
print("\n" + "="*50)
print("非監督式學習結果 (KMeans 分群)")
print("="*50)
print(df[['student_id','cluster']].head())

# -------------------------------
# 5. 分群特徵平均值分析
# -------------------------------
cluster_summary = df.groupby('cluster').mean(numeric_only=True)
print("\n" + "="*50)
print("各群平均特徵")
print("="*50)
print(cluster_summary)
