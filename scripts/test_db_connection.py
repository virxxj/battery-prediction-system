from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="config/.env")

uri = os.getenv('MONGO_URI')

if not uri:
    print("MONGO_URI not found in the environment variables.")
    exit()

print(f"Using MongoDB URI: {uri}")

client = MongoClient(uri, server_api=ServerApi('1'))
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
    
    db = client['battery_monitoring']
    
    collection = db['battery_data']
    
    print(f"Database '{db.name}' and Collection '{collection.name}' are ready to store data.")

except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")
