import os
from dotenv import load_dotenv
load_dotenv(".env", override=True)

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from index.storage import Storage

from openai import AzureOpenAI
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings

from pydantic import BaseModel
from typing import Optional, Dict, List

from utils.utils import format_sources


openai_endpoint = os.getenv("OPENAI_ENDPOINT")
openai_api_version = os.getenv("OPENAI_API_VERSION")
openai_api_key = os.getenv("OPENAI_API_KEY")

chat_deployment = os.getenv("CHAT_DEPLOYMENT")
embedding_deployment = os.getenv("EMBEDDING_DEPLOYMENT")

document_endpoint = os.getenv("DOCUMENT_ENDPOINT")
document_api_key = os.getenv("DOCUMENT_API_KEY")

search_endpoint = os.getenv("SEARCH_ENDPOINT")
search_api_key = os.getenv("SEARCH_API_KEY")
index_name = os.getenv("INDEX_NAME")
semant_config_name = os.getenv("SEMANTIC_CONFIGURATION_NAME")


SEARCH_TYPES = ['text', 'semantic', 'vector', 'semantic_text', 'semantic_vector', 'vector_text','vector_text_semantic']


def get_chatopenai_client():
    """
    Returns an instance of the AzureChatOpenAI client.
    """
    
    return AzureChatOpenAI(
        azure_endpoint=openai_endpoint,
        api_version=openai_api_version,
        api_key=openai_api_key,
        azure_deployment=chat_deployment
    )

def get_openai_client():
    """
    Returns an instance of the AzureOpenAI client.
    """
    
    return AzureOpenAI(
        api_version=openai_api_version,
        azure_endpoint=openai_endpoint,
        api_key=openai_api_key
    )

def get_document_client():
    """
    Returns an instance of the DocumentIntelligenceClient client.
    """

    return DocumentIntelligenceClient(
            endpoint=document_endpoint, 
            credential=AzureKeyCredential(document_api_key)
        )

def get_search_client():
    """
    Returns an instance of the SearchClient client.
    """

    return SearchClient(
        endpoint=search_endpoint,
        credential=AzureKeyCredential(search_api_key),
        index_name=index_name
    )

def get_embeddings_client():
    """
    Returns an instance of the AzureOpenAIEmbeddings client.
    """

    return AzureOpenAIEmbeddings(
        azure_endpoint=openai_endpoint,
        openai_api_version=openai_api_version,
        api_key=openai_api_key,
        model=embedding_deployment
    )

def get_sources_from_container():
    """
    Returns all the blobs contents present in a given container.
    """
    stor = Storage()
    chunks = stor.list_container()
    return chunks

def get_embedding(text):
    """
    Returns the embedding from the OpenAIEmbeddings client.
    """
    client = get_embeddings_client()
    embedding = client.embed_query(text=text)
    return embedding

def retrieve(search_query: str, use_text=True, use_vector=True, use_semantic=False, top=5, knn=10):
    """
    Returns the retrieved results from the Azure Search client.
    """
    client = get_search_client()

    if use_vector:
        search_vector = get_embedding(search_query)

    result = client.search(
            search_query if use_text else None, 
            top=top, 
            vector_queries=[
                VectorizedQuery(vector=search_vector, k_nearest_neighbors=knn, fields="vector")] if use_vector else None,
            query_type="semantic" if use_semantic else None, 
            semantic_configuration_name=semant_config_name if use_semantic else None,
    )
    return result

def get_response(query: str, sys_template: str, messages: Optional[List[Dict]] = [], search_type: Optional[str] = "vector_text", use_all_sources: bool = False, search_kwargs: Optional[Dict] = {}, openai_kwargs: Optional[Dict] = {}, output_schema: Optional[BaseModel] = None, debug: bool = False):
    """
    Returns a response from the OpenAI client.
    """

    assert query and sys_template
    assert search_type in SEARCH_TYPES

    use_text_search = 'text' in search_type
    use_vector_search = 'vector' in search_type
    use_semantic_search = 'semantic' in search_type

    if use_all_sources:
        sources = get_sources_from_container()
    else:
        sources = retrieve(query, use_text=use_text_search, use_vector=use_vector_search, use_semantic=use_semantic_search, **search_kwargs)
    
    sources_formatted, sources_list = format_sources(sources)

    sys_prompt = sys_template.format(sources=sources_formatted)
    sys_prompt = sys_prompt.strip()

    if debug:
        print(sys_prompt)

    sys_msg = {"role": "system", "content": f"{sys_prompt}"}
    user_msg = {"role": "user", "content": f"{query}"}

    chat = [sys_msg] + messages + [user_msg]

    if debug:
        print(chat)
    
    openai_client = get_openai_client()

    completion_func = openai_client.chat.completions.parse if output_schema else openai_client.chat.completions.create

    completion_kwargs = {\
                        "model": chat_deployment,
                        "messages": chat,
                        "temperature": openai_kwargs.get("temperature", 0.1),
                        "top_p": 1.0,
                        "max_tokens": openai_kwargs.get("max_tokens", 1000)
    }
    
    if output_schema:
        completion_kwargs["response_format"] = output_schema

    completion = completion_func(**completion_kwargs)

    if output_schema:
        response = completion.choices[0].message.parsed
    else:
        response = completion.choices[0].message.content

    return response, sources_formatted, sources_list
