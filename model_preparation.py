import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Загружаем предобработанные данные для обучения
train_data = pd.read_csv("train/train_data_scaled.csv")

# Создаем случайный признак
train_data['random_feature'] = np.random.randn(len(train_data))

# Проверяем, какие признаки содержатся в данных после добавления случайного признака
print("Признаки в данных для обучения после добавления случайного признака:")
print(train_data.columns)

# Создаем и обучаем модель
model = LinearRegression()
model.fit(train_data.drop(columns="temperature"), train_data["temperature"])

# Сохраняем обученную модель
joblib.dump(model, "trained_model.pkl")