# database.py
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

DB_SERVER = os.getenv("SQL_DB_SERVER")
DB_PORT = os.getenv("SQL_DB_PORT")
DB_NAME = os.getenv("SQL_DB_NAME")
DB_USER = os.getenv("SQL_DB_USER")
DB_PASSWORD = os.getenv("SQL_DB_PASSWORD")
DB_DRIVER = os.getenv("SQL_DB_DRIVER").replace(" ", "+")  # URL encode spaces

POOL_SIZE = int(os.getenv("SQL_POOL_SIZE", 5))
MAX_OVERFLOW = int(os.getenv("SQL_MAX_OVERFLOW", 10))

# Create the connection string
DATABASE_URL = (
    f"mssql+pyodbc://{DB_USER}:{DB_PASSWORD}@{DB_SERVER}:{DB_PORT}/{DB_NAME}"
    f"?driver={DB_DRIVER}"
)

# Create the SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    pool_size=POOL_SIZE,
    max_overflow=MAX_OVERFLOW,
    echo=False  # Set to False in production
)
