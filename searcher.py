from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

class NeuralSearcher:

    def __init__(self, collection_name, model, port=6333):
        self.collection_name = collection_name
        # Initialize encoder model
        self.model = SentenceTransformer(model, device='cpu')
        # initialize Qdrant client
        self.qdrant_client = QdrantClient(host='localhost', port=port)

    def search(self, text: str, limit=5):
        # Convert text query into vector
        vector = self.model.encode(text).tolist()

        # Use `vector` for search for closest vectors in the collection
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=None,  # We don't want any filters for now
            limit= limit # 5 the most closest results is enough
        )
        # `search_result` contains found vector ids with similarity scores along with the stored payload
        # In this function we are interested in payload only
        payloads = [hit.payload for hit in search_result]
        return payloads
