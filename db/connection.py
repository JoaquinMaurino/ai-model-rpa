import os
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Carga las variables del .env
load_dotenv()

def get_engine():
    try:
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASSWORD")
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT", "5432")
        database = os.getenv("DB_NAME")

        # URL con conexión SSL requerida
        DATABASE_URL = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}?sslmode=require"

        engine = create_engine(DATABASE_URL)
        return engine
    except SQLAlchemyError as e:
        print("❌ Error creando el motor de SQLAlchemy:")
        print(e)
        return None