import os
from dotenv import load_dotenv
import json
import boto3
import numpy as np
from tqdm import tqdm
import csv
from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection
import os
from dotenv import load_dotenv, find_dotenv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.decomposition import PCA


load_dotenv(find_dotenv())

endpoint ="search-search-sgd-rag-tst-xq2prgvmk76osh6de5w6yq5jci.us-east-1.es.amazonaws.com"
region = "us-east-1"
service = "es"

AWS_ACCESS_KEY_ID= os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY= os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_SESSION_TOKEN= os.getenv('AWS_SESSION_TOKEN')

print(f"AWS_ACCESS_KEY_ID : {AWS_ACCESS_KEY_ID} \n AWS_SECRET_ACCESS_KEY : {AWS_SECRET_ACCESS_KEY} \n AWS_SESSION_TOKEN : {AWS_SESSION_TOKEN}")

aws_auth = AWS4Auth( AWS_ACCESS_KEY_ID,  AWS_SECRET_ACCESS_KEY,  region,  service, session_token=AWS_SESSION_TOKEN)

client = OpenSearch(
    hosts = [{'host': endpoint, 'port': 443}],
    http_auth = aws_auth,
    connection_class = RequestsHttpConnection,
    use_ssl = True,
    ssl_show_warn = False, 
    verify_cert = True,
    timeout=10000
)
ef_construction = "ef_construction_650"
index_name = "document_chunks"

def plot_question_distribution(
    question_embeddings: np.ndarray,
    labels: np.ndarray = None,
    methods: list = ("pca", "tsne", "umap"),
    output_dir: str = "./results/question_distribution",
    title_prefix: str = "Questions"
):
    """
    Produce a 1x3 plot of PCA, tSNE, and UMAP for your question embeddings.

    Args:
        question_embeddings:  (n_questions, emb_dim) array
        labels:                optional array of length n_questions (e.g. category codes)
        methods:               which projections to run (subset of "pca","tsne","umap")
        output_dir:            where to save the PNG
        title_prefix:          base title
    """
    os.makedirs(output_dir, exist_ok=True)
    n = question_embeddings.shape[0]

    fig, axes = plt.subplots(1, len(methods), figsize=(5*len(methods), 5))
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        if method.lower() == "pca":
            proj = PCA(n_components=2, random_state=42).fit_transform(question_embeddings)
        elif method.lower() == "tsne":
            perp = min(30, max(2, n - 1))
            proj = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(question_embeddings)
        elif method.lower() == "umap":
            proj = umap.UMAP(n_components=2, random_state=42).fit_transform(question_embeddings)
        else:
            raise ValueError(f"Unknown method {method!r}")

        if labels is None:
            ax.scatter(proj[:, 0], proj[:, 1], s=15, alpha=0.7)
        else:
            ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab20", s=15, alpha=0.7)
        ax.set_title(method.upper())
        ax.set_xticks([])
        ax.set_yticks([])

    n_groups = len(np.unique(labels))
    fig.suptitle(f"{title_prefix} Embedding Distribution — {n_groups} groups", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{title_prefix.replace(' ','_')}_distribution.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved question-distribution plot to {out_path}")

def plot_embeddings(query_embedding, retrieved_embeddings, method="tsne", title="Embedding Projection"):
    all_embeddings = np.vstack([query_embedding] + retrieved_embeddings)
    n_samples = all_embeddings.shape[0]

    # Choose reducer
    if method == "tsne":
        perplexity = min(30, max(2, n_samples - 1))  # Ensure valid perplexity
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    elif method == "umap":
        reducer = umap.UMAP(n_components=2)
    elif method == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Unsupported method")

    reduced = reducer.fit_transform(all_embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[1:, 0], reduced[1:, 1], c='green', label='Retrieved Chunks', s=40)
    plt.scatter(reduced[0, 0], reduced[0, 1], c='blue', label='Query', s=100, marker='x')
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.savefig(f"./results/{method}/{ef_construction}/{title.replace(' ', '_').replace('/', '')}.png")  # Save the plot as a PNG file


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

def queries(question, bedrock, topk=20, text_boost=0.15):
    query_vector = embed_call(bedrock, question)['embedding']
    """query = {
                    "size": topk,
                    "query": {
                        "bool": {
                            "should": [
                                # Text component
                                {
                                    "match": {
                                        "chunk": {
                                            "query": question,
                                            "boost": text_boost
                                        }
                                    }
                                },
                                # Vector component
                                {
                                    "knn": {
                                        "embedding": {
                                            "vector": query_vector,
                                            "k": topk,
                                            "boost": 1.0-text_boost
                                        }
                                    }
                                }
                            ]
                        }
                    }
                }
    
    response = client.search(
        body=query,
        index=index_name,
        request_timeout=100000,
    )"""

    return None, query_vector
    
if __name__ == "__main__":
    bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
    with open('preguntas_especificas.csv', 'r') as f:
        reader = csv.reader(f)
        questions = list(reader)[1:]
    resp_ = []
    question_embeddings = []
    docs = [q[0] for q in questions][:101]                          # list of doc-IDs or paths
    unique_docs = sorted(set(docs))
    doc_to_int = {doc: i for i, doc in enumerate(unique_docs)}
    labels = np.array([doc_to_int[d] for d in docs])
    itr = 0
    for question in tqdm(questions):
        resp, question_embedding = queries(question[1], bedrock)
        question_embeddings.append(question_embedding)
        itr += 1
        if itr > 100:
            break
        """retrieved_embeddings = []
        for hit in resp['hits']['hits']:
            chunk_embedding = hit['_source']['embedding']
            retrieved_embeddings.append(chunk_embedding)

        if len(retrieved_embeddings) > 0:
            plot_embeddings(
                query_embedding=np.array(question_embedding),
                retrieved_embeddings=np.array(retrieved_embeddings),
                method="tsne",  # or "umap" or "pca"
                title=f"Embedding Projection for: {question[1][:50]}..."
            )
            plot_embeddings(
                query_embedding=np.array(question_embedding),
                retrieved_embeddings=np.array(retrieved_embeddings),
                method="umap",  # or "umap" or "pca"
                title=f"Embedding Projection for: {question[1][:50]}..."
            )
            plot_embeddings(
                query_embedding=np.array(question_embedding),
                retrieved_embeddings=np.array(retrieved_embeddings),
                method="pca",  # or "umap" or "pca"
                title=f"Embedding Projection for: {question[1][:50]}..."
            )
        with open('benchmark_less_6p_results_especificas.csv', 'a') as f:
            writer = csv.writer(f)
            for hit in resp['hits']['hits']:
                writer.writerow([question[0], question[1], hit['_source']['chunk'], hit['_score'], hit['_source']['document']])"""
        
    question_embeddings = np.vstack(question_embeddings)
    print(labels)
    plot_question_distribution(
        question_embeddings=question_embeddings,
        labels=labels,               # or your integer label array
        methods=["pca","tsne","umap"],
        output_dir="./results/overview",
        title_prefix="Specific_Questions"
    )

    #open benchmark_less_6p_results.csv and benchmark_less_6p.csv, group by question, and compare if document is the same
    #250 no hyde 0.6235294117647059
    #650 no hyde 0.611764705882353
    with open('benchmark_less_6p_results_especificas.csv', 'r') as f:
        reader = csv.reader(f)
        questions = list(reader)[1:]
    questions = [[question[0], question[1], question[4]] for question in questions]
    questions = list(set(tuple(question) for question in questions))
    compare = {}
    for question in questions:
        if question[1] not in compare:
            compare[question[1]] = {"Real": question[0], "Predicted":[question[2].split('Documentos/')[1]]}
        else:
            compare[question[1]]['Predicted'].append(question[2].split('Documentos/')[1])

    counter = 0
    for question in compare:
        if compare[question]['Real'] in compare[question]['Predicted']:
            counter += 1
    print(f"Accuracy: {counter/len(compare)}")
    