import pandas as pd
from sklearn.preprocessing import StandardScaler

# Загружаем данные для обучения
train_data = pd.read_csv("train/train_data.csv")

# Проводим стандартизацию данных
scaler = StandardScaler()
scaled_data = scaler.fit_transform(train_data)

# Сохраняем предобработанные данные
pd.DataFrame(scaled_data, columns=train_data.columns).to_csv("train/train_data_scaled.csv", index=False)