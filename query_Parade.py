import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

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

model = SentenceTransformer('all-MiniLM-L6-v2')
# === Config ===
PDF_DIR = './Documentos'
conn = psycopg2.connect(
    dbname=database,
    user=postgres,
    password=password,
    host=host,
    port=port
)
cur = conn.cursor()

# Example parameters
search_term = 'keyboard'
embedding_vector = model.encode(search_term).astype(np.float32)

# Run the hybrid query
cur.execute("""
WITH bm25_ranked AS (
    SELECT id, RANK() OVER (ORDER BY bm25_score DESC) AS rank
    FROM (
      SELECT
        id,
        ts_rank(to_tsvector('spanish', text_content), plainto_tsquery(%s)) AS bm25_score
      FROM Test_Luis_pdf
      WHERE text_content @@ plainto_tsquery(%s)
      ORDER BY bm25_score DESC
      LIMIT 20
    ) AS bm25_scored
),
semantic_search AS (
    SELECT id, RANK() OVER (ORDER BY embedding <=> %s::vector) AS rank
    FROM Test_Luis_pdf
    ORDER BY embedding <=> %s::vector
    LIMIT 20
)
SELECT
    COALESCE(semantic_search.id, bm25_ranked.id) AS id,
    COALESCE(1.0 / (60 + semantic_search.rank), 0.0) +
    COALESCE(1.0 / (60 + bm25_ranked.rank), 0.0) AS score,
    Test_Luis_pdf.filename,
    Test_Luis_pdf.text_content
FROM semantic_search
FULL OUTER JOIN bm25_ranked ON semantic_search.id = bm25_ranked.id
JOIN Test_Luis_pdf ON Test_Luis_pdf.id = COALESCE(semantic_search.id, bm25_ranked.id)
ORDER BY score DESC
LIMIT 5;
""", (search_term, search_term, embedding_vector.tolist(), embedding_vector.tolist()))

# Fetch and print results
results = cur.fetchall()
for row in results:
    print(f"ID: {row[0]}, Score: {row[1]:.4f}, Filename: {row[2]}")
    print(f"Snippet: {row[3][:200]}...\n")

cur.close()
conn.close()
