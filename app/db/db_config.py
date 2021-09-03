import os

from databases import DatabaseURL

MAX_CONNECTIONS_COUNT = 10
MIN_CONNECTIONS_COUNT = 10

MONGO_HOST = "localhost"
MONGO_PORT = 27017
MONGO_USER = "admin"
MONGO_PASS = "admin123"
MONGO_DB   = "deepface"

MONGODB_URL = DatabaseURL(
            f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_HOST}:{MONGO_PORT}/{MONGO_DB}"
        )

database_name = MONGO_DB
face_collection_name = "deepface"
