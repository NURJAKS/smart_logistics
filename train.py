import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

print("Загрузка данных...")
df = pd.read_csv('logistic_history.csv')

# Разделяем данные: X - условия, y - результат (задержка)
X = df[['distance_km', 'hour', 'is_rain', 'traffic_score']]
y = df['delay_minutes']

# Делим на обучающую и тестовую выборки (80% на 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Обучение модели (RandomForest)...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Проверяем точность модели (это понадобится для жюри)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Точность модели: алгоритм ошибается в среднем на {mae:.2f} минут(ы).")

# Сохраняем готовую модель
joblib.dump(model, 'model.pkl')
print("Успех! Мозг системы сохранен в файл 'model.pkl'.")
