import os
from dotenv import load_dotenv
load_dotenv(".env", override=True)

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection
)

def connect_to_storage():
    SEARCH_ENDPOINT=os.getenv("SEARCH_ENDPOINT")
    SEARCH_API_KEY=os.getenv("SEARCH_API_KEY")

    STORAGE_CONNECTION_STRING=os.getenv("STORAGE_CONNECTION_STRING")
    STORAGE_API_KEY=os.getenv("STORAGE_API_KEY")

    # Create a data source 
    indexer_client = SearchIndexerClient(endpoint=SEARCH_ENDPOINT, credential=AzureKeyCredential(SEARCH_API_KEY))
    container = SearchIndexerDataContainer(name="es1cont")
    data_source_connection = SearchIndexerDataSourceConnection(
        name="documents",
        type="azureblob",
        connection_string=STORAGE_CONNECTION_STRING,
        container=container
    )
    data_source = indexer_client.create_or_update_data_source_connection(data_source_connection)

    print(f"Data source '{data_source.name}' created or updated")

    return data_source