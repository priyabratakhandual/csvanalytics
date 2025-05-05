from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

def get_collection(collection_name="data"):
    MONGODB_URI = os.getenv("MONGODB_URI")
    client = MongoClient(MONGODB_URI,serverSelectionTimeoutMS=100000)
    db = client["mydatabase"]
    collection = db[collection_name]
    return collection


def clear_collection(collection):
    collection.delete_many({})

def insert_many_records(collection, records):
    collection.insert_many(records)

def get_uploaded_files_count(collection):
    return collection.count_documents({})
