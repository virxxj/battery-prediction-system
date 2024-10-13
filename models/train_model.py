import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
from lstm_model import BatteryLSTM
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load environment variables
load_dotenv(dotenv_path="config/.env")
uri = os.getenv('MONGO_URI')

# Connect to MongoDB
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['battery_monitoring']
collection = db['battery_data']

# Fetch data from MongoDB and preprocess
def fetch_data():
    # Fetch all documents
    data = list(collection.find())
    df = pd.DataFrame(data)
    df = df[['voltage', 'current', 'temperature', 'timestamp']]
    df = df.sort_values('timestamp')
    print(f"Fetched {len(df)} data points.")
    return df[['voltage', 'current', 'temperature']].values

# Prepare data for LSTM
def prepare_data(data, sequence_length=10):
    X = []
    y = []
    if len(data) < sequence_length:
        print("Not enough data to create sequences.")
        return np.array(X), np.array(y)

    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length, 0])  # Predicting the voltage as an example

    return np.array(X), np.array(y)

# Train the LSTM model
def train_model(model, X_train, y_train, num_epochs=100, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        
        # Convert numpy arrays to tensors
        inputs = torch.from_numpy(X_train).float()
        targets = torch.from_numpy(y_train).float()
        
        # Move tensors to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            inputs = inputs.cuda()
            targets = targets.cuda()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Save the model after training
    torch.save(model.state_dict(), "models/battery_lstm_model.pth")
    print("Model training complete and saved!")

if __name__ == "__main__":
    # Fetch and preprocess data
    print("Fetching data from MongoDB...")
    raw_data = fetch_data()
    
    # Normalize data
    if len(raw_data) == 0:
        print("No data available for training.")
        exit()

    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(raw_data)
    
    # Prepare data for LSTM
    sequence_length = 10
    X, y = prepare_data(normalized_data, sequence_length)
    print(f"Data prepared with input shape: {X.shape}, target shape: {y.shape}")
    
    # Check if there is enough data to proceed with training
    if X.shape[0] == 0 or y.shape[0] == 0:
        print("Insufficient data for training. Please collect more data.")
        exit()
    
    # Initialize the model, train it, and save it
    input_size = X.shape[2]  # Number of features (voltage, current, temperature)
    model = BatteryLSTM(input_size=input_size)
    train_model(model, X, y)
