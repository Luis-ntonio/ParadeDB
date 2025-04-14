import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
import numpy as np
from db.connection import model


def create_embedding_table(conn, embedding_dim=384):
    """
    Crea la tabla 'chunks' en PostgreSQL utilizando la extension PGVector.
    Se asume que la extensión 'vector' esta instalada en la base de datos.

    Args:
        conn: Conexión a la base de datos.
        embedding_dim (int): Dimensión del embedding a almacenar.

    Returns:
        None
    """
    cur = conn.cursor()
    # Asegurarse de que la extensión PGVector esté instalada
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    # Crear similarity
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_similarity;")
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")


    # Crear la tabla para almacenar los chunks y sus embeddings
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            name TEXT,
            indexes TEXT,
            text TEXT,
            embedding VECTOR({embedding_dim})
        );
    """)
    conn.commit()
    cur.close()

def insert_embedding_chunks(conn, chunks, indexes, name):
    """
    Para cada chunk, se calcula su embedding y se inserta junto con el texto en la tabla.
    """
    cur = conn.cursor()
    data = []
    for i, chunk in enumerate(chunks):
        # Obtener el embedding como lista de floats
        embedding = model.encode(chunk).tolist()
        data.append((name, indexes[i], chunk, embedding))
    query = "INSERT INTO chunks (name, indexes, text, embedding) VALUES %s"
    execute_values(cur, query, data)
    conn.commit()
    cur.close()