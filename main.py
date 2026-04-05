from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import math
import joblib
import pandas as pd
import os
from datetime import datetime

# Загружаем ML-модель
MODEL_PATH = "model.pkl"
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

app = FastAPI(title="Smart Logistics API")

# URL нашего локального VROOM
VROOM_URL = "http://localhost:13000/"

# Модель того, что мы ожидаем получить от фронтенда
class DeliveryRequest(BaseModel):
    # Список точек в формате [longitude, latitude]
    locations: list[list[float]]

class SimulationRequest(DeliveryRequest):
    # Индекс точки в списке locations, для которой симулируем проблему
    problem_index: int = 1
    # Опциональные параметры (если None - берем реальные данные)
    hour: int | None = None
    is_rain: bool | None = None
    traffic_score: int = 5 # 1-10

def get_current_weather(lat: float, lon: float):
    """
    Запрашивает текущую погоду через Open-Meteo.
    """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=precipitation,weather_code&timezone=auto"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        current = data.get("current", {})
        # weather_code >= 50 означает осадки (дождь, снег и т.д.)
        is_rain = current.get("precipitation", 0) > 0 or current.get("weather_code", 0) >= 50
        return {"is_rain": is_rain, "temp": current.get("temperature_2m")}
    except Exception as e:
        print(f"Ошибка Open-Meteo: {e}")
        return {"is_rain": False, "temp": None}

def calculate_osrm_matrix(locations):
    """
    Получает матрицы времени и расстояния от OSRM.
    """
    coords_str = ";".join([f"{lon},{lat}" for lon, lat in locations])
    url = f"http://router.project-osrm.org/table/v1/driving/{coords_str}?annotations=duration,distance"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Конвертируем в целые числа (VROOM ожидает unsigned int)
        durations = [[int(val) if val is not None else 999999 for val in row] for row in data['durations']]
        distances = data.get('distances', [])
        return durations, distances
    except Exception as e:
        print(f"Ошибка OSRM API: {e}")
        raise HTTPException(status_code=502, detail="Не удалось получить данные от OSRM")

@app.post("/plan-route")
def plan_route(request: DeliveryRequest):
    locations = request.locations
    
    if len(locations) < 2:
        raise HTTPException(status_code=400, detail="Нужно минимум 2 точки (старт и доставка)")

    # 1. Генерируем матрицу через OSRM
    matrix, _ = calculate_osrm_matrix(locations)

    # 2. Формируем запрос для VROOM
    vroom_payload = {
        "vehicles": [
            {
                "id": 1,
                "profile": "custom", # Указываем, что используем свою матрицу
                "start_index": 0,    # Курьер стартует с точки 0
                "end_index": 0       # И возвращается на точку 0 (депо)
            }
        ],
        "jobs": [],
        "matrices": {
            "custom": {
                "durations": matrix
            }
        }
    }

    # Добавляем точки доставки (начиная с 1, т.к. 0 - это старт)
    for i in range(1, len(locations)):
        vroom_payload["jobs"].append({
            "id": i,
            "location_index": i
        })

    # 3. Отправляем математику в движок
    try:
        response = requests.post(VROOM_URL, json=vroom_payload)
        response.raise_for_status()
        vroom_data = response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Ошибка VROOM: {str(e)}")

    # 4. Извлекаем порядок точек из ответа
    steps = vroom_data.get("routes", [{}])[0].get("steps", [])
    
    # Возвращаем красивый ответ
    return {
        "status": "success",
        "optimized_route_indices": [step["location_index"] for step in steps if step["type"] == "job"],
        "total_cost": vroom_data.get("summary", {}).get("cost", 0),
        "vroom_raw_response": vroom_data # Оставим для дебага
    }

@app.post("/simulate-issue")
def simulate_issue(request: SimulationRequest):
    """
    Симулируем проблему, используя ML-модель и РЕАЛЬНЫЕ данные о погоде/времени.
    """
    locations = request.locations
    if len(locations) < 2:
        raise HTTPException(status_code=400, detail="Нужно минимум 2 точки")
    
    if request.problem_index >= len(locations):
        raise HTTPException(status_code=400, detail="problem_index вне диапазона")

    # 1. Получаем реальную матрицу
    matrix, distances = calculate_osrm_matrix(locations)

    # 2. Определяем параметры для модели (авто или ручные)
    # Время
    current_hour = request.hour if request.hour is not None else datetime.now().hour
    
    # Погода (берем для координат проблемной точки)
    prob_lon, prob_lat = locations[request.problem_index]
    if request.is_rain is None:
        weather = get_current_weather(prob_lat, prob_lon)
        final_is_rain = weather["is_rain"]
    else:
        final_is_rain = request.is_rain

    # 3. Предсказание задержки
    predicted_delay_seconds = 0
    if model and distances:
        dist_km = distances[0][request.problem_index] / 1000.0
        
        input_data = pd.DataFrame([{
            'distance_km': dist_km,
            'hour': current_hour,
            'is_rain': int(final_is_rain),
            'traffic_score': request.traffic_score
        }])
        
        delay_minutes = model.predict(input_data)[0]
        predicted_delay_seconds = int(delay_minutes * 60)
        
        # Модифицируем матрицу: задержка на пути к проблемной точке
        matrix[0][request.problem_index] += predicted_delay_seconds

    # 4. Отправляем в VROOM
    vroom_payload = {
        "vehicles": [{"id": 1, "profile": "custom", "start_index": 0, "end_index": 0}],
        "jobs": [{"id": i, "location_index": i} for i in range(1, len(locations))],
        "matrices": {"custom": {"durations": matrix}}
    }

    try:
        response = requests.post(VROOM_URL, json=vroom_payload)
        response.raise_for_status()
        vroom_data = response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    steps = vroom_data.get("routes", [{}])[0].get("steps", [])
    return {
        "status": "incident_simulated",
        "predicted_delay_seconds": predicted_delay_seconds,
        "context": {
            "hour": current_hour,
            "is_rain": final_is_rain,
            "traffic": request.traffic_score,
            "problem_point": locations[request.problem_index]
        },
        "optimized_route_indices": [step["location_index"] for step in steps if step["type"] == "job"],
        "message": f"Задержка {predicted_delay_seconds}с рассчитана для точки {request.problem_index}."
    }
@app.get("/health")
def health_check():
    return {"status": "FastAPI is running!"}
