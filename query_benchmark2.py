from sklearn.metrics import precision_score, recall_score, ndcg_score
import psycopg2
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
import json
import boto3
import torch
import bitsandbytes as bnb
from tqdm import tqdm
import csv

print(torch.cuda.is_available())  # Should return True if GPU is available


load_dotenv()

# === PostgreSQL Config ===
postgres = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")
host = os.getenv("POSTGRES_HOST")
port = int(os.getenv("POSTGRES_PORT", 5432))
database = os.getenv("POSTGRES_DB")

schema = os.getenv("SCHEMA_NAME")
table = os.getenv("TABLE_NAME")

# Initialize Mistral Instruct model for synthetic query generation
device = "cuda" if torch.cuda.is_available() else "cpu"

quantization_config = BitsAndBytesConfig(load_in_8bit=True,
                                         llm_int8_threshold=200.0)

model_name = "mistralai/Mistral-7B-Instruct-v0.3"  # You can replace this with the actual model if different
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically use available GPUs or CPU
    quantization_config=quantization_config,
)

model.generation_config.pad_token_id = tokenizer.pad_token_id



def embed_body(chunk_message : str):
    return json.dumps({
        'inputText' : chunk_message,
        
    })

def embed_call(bedrock : boto3.client, chunk_message : str):
    
    model_id = "amazon.titan-embed-text-v2:0"
    body = embed_body(chunk_message)

    response = bedrock.invoke_model(
        body = body,
        modelId = model_id,
        contentType = 'application/json',
        accept = 'application/json'        
    )    

    return json.loads(response['body'].read().decode('utf-8'))

# Connect to PostgreSQL database
def connect_db():
    conn = psycopg2.connect(
        dbname=database,
        user=postgres,
        password=password,
        host=host,
        port=port
    )
    return conn

# Function to retrieve documents from PostgreSQL database
def get_documents(conn):
    with conn.cursor() as cursor:
        cursor.execute(f"""
            SELECT id, document, embedding FROM {schema}.{table};
        """)
        rows = cursor.fetchall()
    return rows

# Function to generate a synthetic query based on a document
def generate_synthetic_query(document_text, max_length=2000):
    input_text = f"Generate a legal question based on the following document:\n\n{document_text[:1024]}"
    
    # Tokenize the input text with padding and truncation
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
    
    # Set attention mask to indicate which tokens are real text (1) and which are padding (0)
    attention_mask = inputs["attention_mask"]

    # Generate query using the model
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1
    )
    
    # Decode the generated query
    generated_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_query

# Function to compute embeddings for text
def get_embeddings(texts, bedrock):
    embedding_vector = embed_call(bedrock, texts)['embedding']
    return embedding_vector

# Function to retrieve top-k documents using the hybrid search (BM25 + Semantic Search)
def retrieve_top_k_documents(conn, query, embedding, k=10):
    # Construct the hybrid search query (BM25 + semantic search)
    hybrid_search_query = f"""
    WITH bm25_ranked AS (
        SELECT id, RANK() OVER (ORDER BY bm25_score DESC) AS rank
        FROM (
            SELECT
                id,
                ts_rank(to_tsvector('spanish', chunk), plainto_tsquery(%s)) AS bm25_score
            FROM {schema}.{table}
            WHERE chunk @@ plainto_tsquery(%s)
            ORDER BY bm25_score DESC
            LIMIT 20
        ) AS bm25_scored
    ),
    semantic_search AS (
        SELECT id, RANK() OVER (ORDER BY embedding <=> %s::vector) AS rank
        FROM {schema}.{table}
        ORDER BY embedding <=> %s::vector
        LIMIT 20
    )
    SELECT
        COALESCE(semantic_search.id, bm25_ranked.id) AS id,
        COALESCE(1.0 / (60 + semantic_search.rank), 0.0) +
        COALESCE(1.0 / (60 + bm25_ranked.rank), 0.0) AS score,
        {schema}.{table}.document
    FROM semantic_search
    FULL OUTER JOIN bm25_ranked ON semantic_search.id = bm25_ranked.id
    JOIN {schema}.{table} ON {schema}.{table}.id = COALESCE(semantic_search.id, bm25_ranked.id)
    ORDER BY score DESC
    LIMIT %s;
    """
    
    # Execute the query with the embeddings and the query string
    with conn.cursor() as cursor:
        cursor.execute(hybrid_search_query, (query, query, embedding, embedding, k))
        rows = cursor.fetchall()
    
    return rows

# Evaluate Precision@k, Recall@k, NDCG@k
def evaluate_retrieval(query, documents, retrieved_docs, k=10):
    y_true = [1 if doc[0] == retrieved_doc[0] else 0 for doc, retrieved_doc in zip(documents, retrieved_docs)]
    
    # Evaluate Precision and Recall at K
    y_pred = [1 if i < k else 0 for i in range(len(retrieved_docs))]
    
    precision_at_k = precision_score(y_true, y_pred)
    recall_at_k = recall_score(y_true, y_pred)
    
    # Evaluate NDCG@k
    ndcg_at_k = ndcg_score([y_true], [np.array([1 if i < k else 0 for i in range(len(retrieved_docs))])])
    
    return precision_at_k, recall_at_k, ndcg_at_k

# Test the retrieval performance for synthetic queries
def test_retrieval_for_queries(conn, synthetic_queries, documents, k=10, bedrock = None):
    for query in synthetic_queries:
        # First, generate the query embedding for semantic search
        query_embedding = get_embeddings(query, bedrock)

        # Retrieve the top-k documents using the hybrid search
        retrieved_docs = retrieve_top_k_documents(conn, query, query_embedding, k)

        # Evaluate using precision, recall, and NDCG
        precision, recall, ndcg = evaluate_retrieval(query, documents, retrieved_docs, k)
        
        print(f"Query: {query[:]}...")  # Print a short preview of the query
        print(f"Precision@{k}: {precision}")
        print(f"Recall@{k}: {recall}")
        print(f"NDCG@{k}: {ndcg}")
        print("-" * 50)


# Example function call
def main():
    bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
    conn = connect_db()  # Connect to PostgreSQL
    documents = get_documents(conn)  # Retrieve documents from DB
    synthetic_queries = []
    document_query = []
    it = 0
    for _, document, _ in tqdm(documents):
        query = generate_synthetic_query(document)
        synthetic_queries.append(query)  # Store generated query
        document_query.append([document, synthetic_queries])
        it += 1
        if it == 10:
            break
        # Save synthetic queries to a CSV file
    with open('queries.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Document'])  # Write header
        csvwriter.writerow(['Query'])  # Write header
        for doc, query in document_query:
            csvwriter.writerow([doc, query])  # Write each query
    
    # Test retrieval performance for the generated synthetic queries
    test_retrieval_for_queries(conn, synthetic_queries, documents, bedrock=bedrock)

    conn.close()

if __name__ == "__main__":
    main()
