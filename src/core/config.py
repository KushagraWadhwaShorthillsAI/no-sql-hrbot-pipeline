import os
from dotenv import load_dotenv

# Load MongoDB connection details from .env file
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "resume_db")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "resumes")
