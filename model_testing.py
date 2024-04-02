import pandas as pd
import joblib
import numpy as np

# Загружаем предобученную модель
model = joblib.load("trained_model.pkl")

# Загружаем данные для тестирования
test_data = pd.read_csv("test/test_data.csv")

# Добавляем отсутствующий признак "random_feature"
test_data['random_feature'] = np.random.randn(len(test_data))

# Предсказываем значения на тестовых данных
predictions = model.predict(test_data.drop(columns="temperature"))

# Выводим предсказанные значения
print("Предсказанные значения:")
print(predictions)