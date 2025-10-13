import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection
)

import json


class Storage():
    """
        Add index data source connection to Blob Storage and upload json chunks
    """

    def __init__(self, title_field: str = "header"):
        self.search_endpoint = os.getenv("SEARCH_ENDPOINT")
        self.search_api_key = os.getenv("SEARCH_API_KEY")

        self.index_name = os.getenv("INDEX_NAME")
        self.data_source_name = os.getenv("DATA_SOURCE_NAME")

        self.storage_name = os.getenv("STORAGE_NAME")
        self.storage_account_url = f"https://{self.storage_name}.blob.core.windows.net"
        self.storage_api_key = os.getenv("STORAGE_API_KEY")
        self.storage_string = os.getenv("STORAGE_CONNECTION_STRING")
        self.container_name = os.getenv("CONTAINER_NAME")

        self.title_field = title_field
    
    def get_indexer_client(self):
        indexer_client = SearchIndexerClient(
            endpoint=self.search_endpoint, credential=AzureKeyCredential(self.search_api_key)
            )
        return indexer_client

    def get_storage_client(self):
        blob_service_client = BlobServiceClient(
            account_url=self.storage_account_url, credential=self.storage_api_key
        )
        return blob_service_client
    
    def get_blob_client(self, storage_client, file_name):
        return storage_client.get_blob_client(container=self.container_name, blob=file_name)

    def connect_to_container(self):
        indexer_client = self.get_indexer_client()
        container = SearchIndexerDataContainer(name=self.container_name)

        data_source_connection = SearchIndexerDataSourceConnection(
            name=self.data_source_name,
            type="azureblob",
            connection_string=self.storage_string,
            container=container
        )

        # Create a data source
        data_source = indexer_client.create_or_update_data_source_connection(data_source_connection) # add try-except
        print(f"Data source '{data_source.name}' created or updated")

        indexer_client.close() # add try-except

    def erase_container(self):
        storage_client = self.get_storage_client()
        container_client = storage_client.get_container_client(self.container_name)

        blob_list = container_client.list_blobs()

        for blob in blob_list: # add try-except
            blob_name = blob.name
            blob_client = self.get_blob_client(storage_client=storage_client, file_name=blob_name)
            blob_client.delete_blob()
            blob_client.close()
        
        print(f"All blobs in container '{self.container_name}' deleted.")

        container_client.close()
        storage_client.close() 

    def upload_to_container(self, data, erase_container=False, overwrite=True):
        if erase_container:
            self.erase_container()

        blob_service_client = self.get_storage_client()
        
        for i, chunk in enumerate(data):
            local_file_name = chunk[self.title_field] + f"_{i+1}"
            blob_client = self.get_blob_client(storage_client=blob_service_client, file_name=local_file_name)
            
            try:
                blob_client.upload_blob(json.dumps(chunk), overwrite=overwrite)
                print("\n   Uploaded to Azure Storage as blob: " + local_file_name)
            except Exception as e:
                print(f"Could not write + {local_file_name} ({e})")

            blob_client.close() # add try-except
        
        blob_service_client.close() # add try-except
