import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()


def create_conn():
    conn = psycopg2.connect(
        dbname="amber",
        user="postgres",
        password="1234",
        host="localhost"
    )
    return conn

# Configuración del modelo para obtener embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # Este modelo genera vectores de dimensión 384