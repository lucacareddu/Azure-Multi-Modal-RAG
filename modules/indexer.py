import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndexer,
    FieldMapping
)


class Indexer():
    """
        Connect index to data source and skillset
    """

    def __init__(self):
        self.search_endpoint = os.getenv("SEARCH_ENDPOINT")
        self.search_api_key = os.getenv("SEARCH_API_KEY")

        self.indexer_name=os.getenv("INDEXER_NAME")
        self.index_name = os.getenv("INDEX_NAME")
        self.data_source_name = os.getenv("DATA_SOURCE_NAME")
        # self.skillset_name = os.getenv("SKILLSET_NAME")

    def get_indexer_client(self):
        indexer_client = SearchIndexerClient(
            endpoint=self.search_endpoint, credential=AzureKeyCredential(self.search_api_key)
            )
        return indexer_client
    
    def build_indexer(self):
        indexer_parameters = {"configuration": {"parsingMode": "json"}}

        indexer = SearchIndexer(  
            name=self.indexer_name,  
            description="Indexer built from the Python SDK",  
            target_index_name=self.index_name,  
            data_source_name=self.data_source_name,
            # skillset_name=self.skillset_name, # not needed this time
            # Map the metadata_storage_name field to the title field in the index to display the PDF title in the search results  
            # field_mappings=[FieldMapping(source_field_name=field, target_field_name=field) for field in ['header', 'raw_content', 'format_content', 'page', 'source', 'url', 'vector']],
            parameters=indexer_parameters # this replaces the field mappings
        )  

        indexer_client = self.get_indexer_client()
        try:
            indexer_client.delete_indexer(self.indexer_name)
        except:
            print(f"Could not delete the indexer '{self.indexer_name}'.")

        # Create the indexer  
        result = indexer_client.create_or_update_indexer(indexer) # add try-except

        print(f'{result.name} is created and running. Give the indexer a few minutes before running a query.')

        indexer_client.close() # add try-except
