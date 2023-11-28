import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from pprint import pprint
from qdrant_client.http.models import Distance, VectorParams
import argparse
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def init_qdrant(port=6333):
    client = QdrantClient("localhost", port=6333)


    client.recreate_collection(
        collection_name="Product Infos",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    return client

def parse_args():
    # parse the input file name and the port number
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6333)
    parser.add_argument('--input_file', type=str, default='bigBasketProducts.csv')
    return parser.parse_args()

def get_embedding_dict(file_name):
    # Define the construct_sentence function
    def construct_sentence(row):
        return f'{row.product}, Category is {row.category}, {row.sub_category}'
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_name)
    # Create a list of sentences using the construct_sentence function
    sentences = []
    indices_dict = {}
    for i, row in enumerate(df.itertuples()):
        sentence = construct_sentence(row)
        sentences.append(sentence)
        indices_dict[sentence] = i
    
    # Load the sentence transformer model
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    batch_size = 100
    num_batches = len(sentences) // batch_size + 1

    encoded_embeddings_dict = {}

    # Process the sentences in batches
    for i in tqdm(range(num_batches)):
        batch = sentences[i*batch_size : min((i+1)*batch_size, len(sentences))]
        
        # Encode the batch of sentences
        batch_embeddings = model.encode(batch)
        
        # Add the batch of encoded embeddings to the dictionary
        for j, sentence in enumerate(batch):
            encoded_embeddings_dict[sentence] = batch_embeddings[j]
    return encoded_embeddings_dict, indices_dict


def upload_to_qdrant(client, embedding_dict):
    # Upload the data to Qdrant
    batch_size = 1000
    num_batches = len(embedding_dict) // batch_size + 1
    keys = list(embedding_dict.keys())
    values = list(embedding_dict.values())

    for j in range(num_batches):
        PointStructs = []
        batch_keys = keys[j*batch_size : min((j+1)*batch_size, len(keys))]
        batch_values = values[j*batch_size : min((j+1)*batch_size,len(values))]
        for i, (sentence, embedding) in enumerate(zip(batch_keys, batch_values)):
            PointStructs.append(PointStruct(
                        id=i,
                        vector=embedding,
                        payload={"sentence": sentence}
                        ))
        client.upsert(collection_name="Product Infos", 
                        wait = True,
                        points = PointStructs)


def setup_qdrant(port, input_file):
    
    client = init_qdrant(port)
    print(f'Connected to qdrant on port {port}')
    
    print(f'Creating embeddings for {input_file}')
    embedding_dict, indices_dict = get_embedding_dict(input_file)
    
    print('Embeddings created, Uploading to qdrant')
    upload_to_qdrant(client, embedding_dict)
    
    return indices_dict

if __name__ == '__main__':
    setup_qdrant(6333, 'bigBasketProducts.csv')