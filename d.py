import os
import fitz  # PyMuPDF
import psycopg2
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

model = SentenceTransformer('all-MiniLM-L6-v2')
# Load environment variables from .env file
load_dotenv()

# === PostgreSQL Config ===
postgres = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")
host = os.getenv("POSTGRES_HOST")
port = int(os.getenv("POSTGRES_PORT", 5432))
database = os.getenv("POSTGRES_DB")

# === Schema and Table Details ===
schema_name = os.getenv("SCHEMA_NAME", "public")
table_name = os.getenv("TABLE_NAME", "test_table")

conn = psycopg2.connect(
    dbname=database,
    user=postgres,
    password=password,
    host=host,
    port=port
)
cur = conn.cursor()

def select_all():
    select_query = f"SELECT document FROM {schema_name}.{table_name};"
    cur.execute(select_query)
    rows = cur.fetchall()
    for row in rows:
        print(row)

select_all()