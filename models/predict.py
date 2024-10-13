import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
from lstm_model import BatteryLSTM
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

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
    print(f"Fetched {len(df)} data points.")
    return df

def prepare_data(data, sequence_length=10):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length, 0])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    print("Fetching data from MongoDB...")
    df = fetch_data()
    raw_data = df[['voltage', 'current', 'temperature']].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(raw_data)
    sequence_length = 10
    X, y_actual = prepare_data(normalized_data, sequence_length)

    input_size = X.shape[2]
    model = BatteryLSTM(input_size=input_size)
    model.load_state_dict(torch.load("models/battery_lstm_model.pth"))

    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(X).float()
        predictions = model(inputs).numpy()

    predicted_voltages = scaler.inverse_transform(np.hstack((predictions, np.zeros((predictions.shape[0], 2)))))[:, 0]
    actual_voltages = scaler.inverse_transform(np.hstack((y_actual.reshape(-1, 1), np.zeros((y_actual.shape[0], 2)))))[:, 0]

    plt.figure(figsize=(10, 5))
    plt.plot(actual_voltages, label="Actual Voltages")
    plt.plot(predicted_voltages, label="Predicted Voltages")
    plt.title("Actual vs. Predicted Voltages")
    plt.xlabel("Time Step")
    plt.ylabel("Voltage")
    plt.legend()
    plt.show()
