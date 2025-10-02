import os
from dotenv import load_dotenv
load_dotenv(".env", override=True)

from azure.core.credentials import AzureKeyCredential

from azure.search.documents.indexes.models import (
    SearchIndexer,
    FieldMapping
)

from azure.search.documents.indexes import SearchIndexerClient


def get_indexer():
    SEARCH_ENDPOINT=os.getenv("SEARCH_ENDPOINT")
    SEARCH_API_KEY=os.getenv("SEARCH_API_KEY")
    SEARCH_NAME=os.getenv("INDEX_NAME")

    INDEXER_NAME=os.getenv("INDEXER_NAME")
    # SKILLSET_NAME=os.getenv("SKILLSET_NAME")

    indexer_parameters = {"configuration": {"parsingMode": "json"}}

    indexer = SearchIndexer(  
        name=INDEXER_NAME,  
        description="Indexer to index documents and generate embeddings",  
        # skillset_name=SKILLSET_NAME,  
        target_index_name=SEARCH_NAME,  
        data_source_name="documents",
        # Map the metadata_storage_name field to the title field in the index to display the PDF title in the search results  
        # field_mappings=[FieldMapping(source_field_name=field, target_field_name=field) for field in ['header', 'raw_content', 'format_content', 'paragraph_number', 'page', 'source', 'vector']],
        parameters=indexer_parameters
    )  

    # Create and run the indexer  
    indexer_client = SearchIndexerClient(endpoint=SEARCH_ENDPOINT, credential=AzureKeyCredential(SEARCH_API_KEY))  
    try:
        indexer_client.delete_indexer(INDEXER_NAME)
    except:
        print("Could not delete indexer.")
        pass
    indexer_result = indexer_client.create_or_update_indexer(indexer)  

    print(f' {INDEXER_NAME} is created and running. Give the indexer a few minutes before running a query.')
    
    return indexer_result