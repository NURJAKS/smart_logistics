import pandas as pd
import numpy as np

print("Начинаю генерацию данных...")

# Генерируем 1000 фейковых доставок
num_records = 1000
np.random.seed(42)

# Входные параметры (Фичи)
distances = np.random.uniform(1.0, 25.0, num_records) # Расстояние от 1 до 25 км
hours = np.random.randint(8, 23, num_records)         # Время с 08:00 до 22:00
is_rain = np.random.choice([0, 1], num_records, p=[0.7, 0.3]) # 1 - дождь (30% случаев), 0 - ясно
traffic_score = np.random.randint(1, 11, num_records) # Пробки от 1 до 10 баллов

# Целевая переменная (Задержка в минутах)
delays = []
for i in range(num_records):
    delay = 0
    # Логика: если дождь, добавляем от 10 до 20 минут задержки
    if is_rain[i] == 1:
        delay += np.random.randint(10, 21)
    
    # Логика: если час пик (утро или вечер) и пробки > 6, добавляем сильную задержку
    if (hours[i] in [8, 9, 17, 18, 19]) and traffic_score[i] > 6:
        delay += np.random.randint(15, 30)
        
    # Добавляем случайный фактор улицы (кто-то припарковался криво и т.д.)
    delay += np.random.randint(0, 6)
    delays.append(delay)

# Собираем все в таблицу
df = pd.DataFrame({
    'distance_km': distances,
    'hour': hours,
    'is_rain': is_rain,
    'traffic_score': traffic_score,
    'delay_minutes': delays
})

# Сохраняем в файл
df.to_csv('logistic_history.csv', index=False)
print("Готово! Файл 'logistic_history.csv' успешно создан.")
