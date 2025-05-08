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

# === PDF Directory ===
PDF_DIR = './Documentos'

# === Connect ===
conn = psycopg2.connect(
    dbname=database,
    user=postgres,
    password=password,
    host=host,
    port=port
)
cur = conn.cursor()

delete_table_query = f"DROP TABLE IF EXISTS {schema_name}.{table_name};"
try:
    cursor = conn.cursor()
    cursor.execute(delete_table_query)
    conn.commit()
    print(f"Table '{table_name}' deleted successfully from schema '{schema_name}'!")
except psycopg2.Error as e:
    print(f"Error: {e}")

def create_table():
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} (
        id SERIAL PRIMARY KEY,
        document TEXT,
        page INTEGER,
        chunk TEXT,
        tsv tsvector,
        embedding vector(1024),
        upload_datetime TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    try:
        connection = psycopg2.connect(
            dbname=database,
            user=postgres,
            password=password,
            host=host,
            port=port
        )
        cursor = connection.cursor()
        cursor.execute(create_table_query)
        connection.commit()

        # Create tsvector GIN index
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_tsv
            ON {schema_name}.{table_name}
            USING GIN(tsv);
        """)

        # Create pgvector index (optional, for ANN search)
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_embedding
            ON {schema_name}.{table_name}
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)

        connection.commit()
        print(f"Table '{table_name}' created successfully in schema '{schema_name}'!")
    except psycopg2.Error as e:
        print(f"Error: {e}")
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'connection' in locals() and connection:
            connection.close()
            print("Connection closed.")


def extract_text(filepath):
    try:
        doc = fitz.open(filepath)
        return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"Failed to extract {filepath}: {e}")
        return ""


def insert_pdf_data():
    for filename in tqdm(os.listdir(PDF_DIR)):
        if filename.lower().endswith('.pdf'):
            full_path = os.path.join(PDF_DIR, filename)
            text = extract_text(full_path)

            with open(full_path, 'rb') as f:
                file_data = f.read()

            embedding = model.encode(text).tolist()

            cur.execute(f"""
                INSERT INTO {schema_name}.{table_name} (filename, text_content, tsv, file_data, embedding)
                VALUES (%s, %s, to_tsvector('spanish', %s), %s, %s)
            """, (filename, text, text, psycopg2.Binary(file_data), embedding))


# === Run ===
create_table()
#insert_pdf_data()

conn.commit()
cur.close()
conn.close()
