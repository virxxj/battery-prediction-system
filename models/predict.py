import torch
import numpy as np
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from lstm_model import BatteryLSTM

load_dotenv(dotenv_path="config/.env")
uri = os.getenv('MONGO_URI')

client = MongoClient(uri, server_api=ServerApi('1'))
db = client['battery_monitoring']
collection = db['battery_data']

def fetch_data():
    data = list(collection.find())
    df = pd.DataFrame(data)
    df = df[['voltage', 'current', 'temperature', 'timestamp']]
    df = df.sort_values('timestamp')
    return df[['voltage', 'current', 'temperature']].values

def prepare_data(data, sequence_length=10):
    X = []
    if len(data) < sequence_length:
        print("Not enough data to create sequences.")
        return np.array(X)

    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])

    return np.array(X)

def predict():
    print("Fetching data from MongoDB...")
    raw_data = fetch_data()

    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(raw_data)

    sequence_length = 10
    X = prepare_data(normalized_data, sequence_length)
    
    if X.shape[0] == 0:
        print("No data available for prediction.")
        return

    X_tensor = torch.from_numpy(X).float()

    input_size = X.shape[2]
    model = BatteryLSTM(input_size=input_size)
    model.load_state_dict(torch.load("models/battery_lstm_model.pth"))
    model.eval()

    with torch.no_grad():
        predictions = model(X_tensor)
        predicted_voltages = predictions.numpy()
        print(f"Predicted Voltages: {predicted_voltages}")

if __name__ == "__main__":
    predict()
