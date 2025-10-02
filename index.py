import os
import json

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticSearch,
    SemanticField
)

load_dotenv(".env", override=True)

endpoint = os.getenv("SEARCH_ENDPOINT")
api_key = os.getenv("SEARCH_API_KEY")


credential = AzureKeyCredential(api_key)
azure_search_service_endpoint = endpoint

def get_search_index_client(search_index_name):

    return SearchIndexClient(
        endpoint=azure_search_service_endpoint, 
        index_name=search_index_name, 
        credential=credential
    )

# create search index

def create_search_index(search_index_name):

    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            sortable=True,
            filterable=True,
            facetable=True,
        ),
        SearchableField(name="header", type=SearchFieldDataType.String),
        SearchableField(name="raw_content", type=SearchFieldDataType.String, facetable=True),
        SearchableField(name="format_content", type=SearchFieldDataType.String),
        SearchableField(name="paragraph", type=SearchFieldDataType.Int32),
        SearchableField(name="page", type=SearchFieldDataType.Int32),
        SearchableField(name="source", type=SearchFieldDataType.String),
        SearchField(name="vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536, 
            vector_search_profile_name="myHnswProfile",
        ),
    ]

    # Configure the vector search configuration  
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw"
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
            )
        ]
    )

    semantic_config = SemanticConfiguration(
        name="my-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="header"),
            content_fields=[SemanticField(field_name="format_content")],
            keywords_fields=[SemanticField(field_name="page"), SemanticField(field_name="paragraph")],
        )
    )

    # Create the semantic settings with the configuration
    semantic_search = SemanticSearch(configurations=[semantic_config])
    # Create the search index with the semantic settings
    search_index = SearchIndex(name=search_index_name, fields=fields,
                        vector_search=vector_search, semantic_search=semantic_search)
    result = get_search_index_client(search_index_name).create_or_update_index(search_index)
    print(f' {result.name} created')

def index_exists(client, index_name):
    indexes = client.list_index_names()
    return index_name in indexes

def delete_index_if_exists(search_index_name):
    client = get_search_index_client(search_index_name)
    try:
        if index_exists(client, search_index_name):
            print(f"Index '{search_index_name}' exists.")
            client.delete_index(search_index_name)
            print(f"Index '{search_index_name}' deleted.")
        else:
            print(f"Index '{search_index_name}' does not exist.")
    except Exception as e:
        print(f"Error deleting index: {e}")


def get_search_client(search_index_name):
    return  SearchClient(
        endpoint=endpoint,
        index_name=search_index_name,
        credential=credential
    )


def upload_chunk_document(filepath, search_index_name):
    search_client = get_search_client(search_index_name)
    # filename = os.path.basename(filepath)

    # if filename.endswith('.json'):
    #     with open(filepath, 'r') as file:
    #         document = json.load(file)
    #         print(f"Uploading {filename} to Azure Search Index...")

    #         result = search_client.upload_documents(documents=document)
    #         print(f"Upload of {filename} succeeded: { result[0].succeeded }")

    result = search_client.upload_documents(documents=filepath)
    print("Upload of '{}'.. succeeded? {}".format(filepath["header"], result[0].succeeded))