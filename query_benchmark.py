import csv
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from tqdm import tqdm

# Load environment variables
load_dotenv()

# === PostgreSQL Config ===
postgres = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")
host = os.getenv("POSTGRES_HOST")
port = int(os.getenv("POSTGRES_PORT", 5432))
database = os.getenv("POSTGRES_DB")

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname=database,
    user=postgres,
    password=password,
    host=host,
    port=port
)
cur = conn.cursor()

# Input and output file paths
input_csv = './benchmark_less_6p.csv'
output_csv = './top1_documents.csv'

# Process each question in the CSV
with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
    reader = csv.DictReader(infile)
    writer = csv.writer(outfile)
    writer.writerow(['Question', 'Top1_Document', 'Score'])  # Output headers

    for row in tqdm(reader):
        question = row['Questions']
        embedding_vector = model.encode(question).astype(np.float32)

        # Execute the hybrid query
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
            Test_Luis_pdf.filename
        FROM semantic_search
        FULL OUTER JOIN bm25_ranked ON semantic_search.id = bm25_ranked.id
        JOIN Test_Luis_pdf ON Test_Luis_pdf.id = COALESCE(semantic_search.id, bm25_ranked.id)
        ORDER BY score DESC
        LIMIT 1;
        """, (question, question, embedding_vector.tolist(), embedding_vector.tolist()))

        # Fetch the top result
        result = cur.fetchone()
        if result:
            top_document = result[2]
            score = result[1]
            writer.writerow([question, top_document, score])
        else:
            writer.writerow([question, None, None])

# Close the database connection
cur.close()
conn.close()