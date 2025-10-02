import uuid
import os
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.ai.inference import EmbeddingsClient

load_dotenv(".env", override=True)

endpoint = os.getenv("EMBEDDING_ENDPOINT")
api_key = os.getenv("EMBEDDING_API_KEY")
embeddings_model_deployment = os.getenv("EMBEDDING_DEPLOYMENT")

def get_client():

    client = EmbeddingsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key)
    )
    return client

def get_embeddings_vector(text):


    response = get_client().embed(
    input=text,
    model=embeddings_model_deployment
    )

    embedding = response.data[0].embedding

    return embedding

def get_chunk_object(paragraph: dict)-> dict:
    vector = get_embeddings_vector(paragraph["format_content"])

    return {
        "id": str(uuid.uuid4()),
        **paragraph,
        'vector': vector
    }
