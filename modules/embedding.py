import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from azure.core.credentials import AzureKeyCredential
from azure.ai.inference import EmbeddingsClient

from typing import List, Dict

import tiktoken


class Embedder():
    """
        Embed text into vector space
    """

    def __init__(self, format_content_field: str = "format_content"):
        self.endpoint = os.getenv("EMBEDDING_ENDPOINT")
        self.api_key = os.getenv("EMBEDDING_API_KEY")
        self.model_deployment = os.getenv("EMBEDDING_DEPLOYMENT")

        self.format_content_field = format_content_field

    def get_client(self):
        client = EmbeddingsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key)
        )
        return client
    
    def tokens_number_test(self, paragraphs: List[Dict], max_tokens=1000):
        tokenizer = tiktoken.get_encoding(encoding_name="cl100k_base")
        tokens = [len(tokenizer.encode(p[self.format_content_field], disallowed_special=())) > max_tokens for p in paragraphs]

        return "succeded" if not any(tokens) else "failed"

    def get_embedding_vector(self, text: str):
        embeddings_client = self.get_client()
        response = embeddings_client.embed(
            input=text,
            model=self.model_deployment
        )

        embeddings_client.close() # add try-except

        embedding = response.data[0].embedding
        return embedding

    def get_chunk_object(self, paragraph: Dict) -> Dict:
        vector = self.get_embedding_vector(paragraph[self.format_content_field])

        chunk = {
                **paragraph,
                'vector': vector
            }
        
        return chunk
