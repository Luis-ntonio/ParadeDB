
import boto3
import os
import fitz  # PyMuPDF
import psycopg2
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from query_benchmark import embed_body, embed_call

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

bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')


def select_all():
        
    select_query = f"""
        SELECT document, chunk
            FROM Test_Luis_pdf
    """
    #q = "¿Cuál es el Código de Cuenta Interbancario (CCI) proporcionado para el abono de pagos?"
    #e = embed_call(bedrock, q)['embedding']
    cur.execute(select_query)#,(e, e))
    rows = cur.fetchall()
    for row in rows:
        print(row)
    

select_all()