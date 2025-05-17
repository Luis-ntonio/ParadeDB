import numpy as np
import matplotlib.pyplot as plt
from opensearchpy import OpenSearch
from sklearn.manifold import TSNE
import umap.umap_ as umap
import os
from tqdm import tqdm
from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection
from dotenv import load_dotenv, find_dotenv
import time 
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
    timeout=1000
)

index_name = "document_chunks"

# Step 2: Extract embeddings and metadata (e.g., page)
def scroll_all_embeddings(index_name, scroll_duration="2m", batch_size=5):
    all_embeddings = []
    all_documents = []
    all_labels = []

    query = {
        "size": batch_size,
        "_source": ["embedding", "document", "label"],
        "query": {"match_all": {}}
    }
    total_hits = 0
    # Initial request
    response = client.search(index=index_name, body=query, scroll=scroll_duration)
    scroll_id = response["_scroll_id"]
    hits = response["hits"]["hits"]
    total_hits += len(hits)
    diction = {}
    while len(hits) > 0:
        for hit in tqdm(hits):
            source = hit["_source"]
            if source["label"] not in diction:
                diction[source["label"]] = 1
            else:
                diction[source["label"]] += 1
            if "embedding" in source:
                all_embeddings.append(source["embedding"])
                all_documents.append(source.get("document", -1))
                all_labels.append(source.get("label", -1).replace('\n', '').replace('\t','').replace("    ", '').split(">")[1].split("<")[0])

        # Fetch next batch
        response = client.scroll(scroll_id=scroll_id, scroll=scroll_duration)
        scroll_id = response["_scroll_id"]
        hits = response["hits"]["hits"]
        total_hits += len(hits)
    # Clear scroll context
    client.clear_scroll(scroll_id=scroll_id)
    print(f"Total hits: {total_hits}")
    return np.array(all_embeddings), np.array(all_documents), np.array(all_labels)


# Step 3: Visualize with TSNE and UMAP
def visualize_embeddings(embeddings, labels, title_suffix=""):
    # -- factor string labels into integers 0,1,2,... --
    unique_docs = np.unique(labels)
    doc_to_int = {doc:i for i,doc in enumerate(unique_docs)}
    numeric_labels = np.array([doc_to_int[doc] for doc in labels])

    fig, axs = plt.subplots(1, 2, figsize=(18, 8))

    # TSNE
    tsne = TSNE(n_components=2, perplexity=30)
    reduced_tsne = tsne.fit_transform(embeddings)
    sc1 = axs[0].scatter(
        reduced_tsne[:, 0], reduced_tsne[:, 1],
        c=numeric_labels,        # now numeric!
        cmap='viridis', s=10
    )
    legend_labels = {v: k for k, v in doc_to_int.items()}
    axs[0].legend(handles=[plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=sc1.cmap(sc1.norm(i)), 
                                  label=legend_labels[i]) for i in range(len(unique_docs))])
    axs[0].set_title("TSNE Visualization " + title_suffix)
    fig.colorbar(sc1, ax=axs[0], ticks=range(len(unique_docs)))

    # UMAP
    reducer = umap.UMAP(n_components=2)
    reduced_umap = reducer.fit_transform(embeddings)
    sc2 = axs[1].scatter(
        reduced_umap[:, 0], reduced_umap[:, 1],
        c=numeric_labels,        # also numeric here
        cmap='viridis', s=10
    )
    axs[1].set_title("UMAP Visualization " + title_suffix)
    fig.colorbar(sc2, ax=axs[1], ticks=range(len(unique_docs)))

    plt.tight_layout()
    output_file = f"visualization_{title_suffix.replace(' ', '_')}.png"
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")
    plt.close()

# Step 4: Run the workflow
if __name__ == "__main__":
    start_time = time.time()
    print("Fetching embeddings from OpenSearch...")
    embeddings, documents, labels = scroll_all_embeddings(index_name=index_name)
    
    if embeddings.size == 0:
        print("No embeddings found in the index.")
    else:
        print(f"Found {len(embeddings)} embeddings.")
        visualize_embeddings(embeddings, labels,  title_suffix=f"(Index: {index_name})")
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")