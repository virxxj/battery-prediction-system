from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv
import random
import time

load_dotenv(dotenv_path="config/.env")

uri = os.getenv('MONGO_URI')

client = MongoClient(uri, server_api=ServerApi('1'))
db = client['battery_monitoring']
collection = db['battery_data']

def generate_sensor_data():
    data = {
        "voltage": round(random.uniform(3.0, 4.2), 2),
        "current": round(random.uniform(0.1, 1.5), 2),
        "temperature": round(random.uniform(20, 40), 2),
        "timestamp": time.time()
    }
    return data

def insert_data():
    sensor_data = generate_sensor_data()
    result = collection.insert_one(sensor_data)
    print(f"Data inserted with ID: {result.inserted_id}")

if __name__ == "__main__":
    try:
        print("Starting data ingestion...")
        max_data_points = 100
        for _ in range(max_data_points):
            insert_data()
            time.sleep(2) 
        print(f"Inserted {max_data_points} data points.")
    except Exception as e:
        print(f"Error during data ingestion: {e}")