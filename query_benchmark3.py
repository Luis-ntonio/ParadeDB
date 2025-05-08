import os
from dotenv import load_dotenv
import json
import boto3
from tqdm import tqdm
import csv
from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection
import os
from dotenv import load_dotenv, find_dotenv

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
    ssl_show_warn = True, 
    verify_cert = True,
    timeout=60
)

index_name = "document_chunks"

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

def queries(question, bedrock, topk=5, text_boost=0.5):
    query_vector = embed_call(bedrock, question)['embedding']
    query = {
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
        index=index_name
    )

    return response
    
if __name__ == "__main__":
    """bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
    with open('benchmark_less_6p.csv', 'r') as f:
        reader = csv.reader(f)
        questions = list(reader)[1:]
    resp_ = []
    for question in tqdm(questions):
        tmp = []
        resp = queries(question[1], bedrock)

        with open('benchmark_less_6p_results.csv', 'a') as f:
            writer = csv.writer(f)
            for hit in resp['hits']['hits']:
                writer.writerow([question[0], question[1], hit['_source']['chunk'], hit['_score'], hit['_source']['document']])"""
    #open benchmark_less_6p_results.csv and benchmark_less_6p.csv, group by question, and compare if document is the same
    with open('benchmark_less_6p_results.csv', 'r') as f:
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
    print(compare)
    print(f"Accuracy: {counter/len(compare)}")
    