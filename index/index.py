import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
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

import json


class Index():
    """
        Build index and upload json chunks
    """

    def __init__(self, title_field: str = "header", use_vector: bool = True):
        self.search_endpoint = os.getenv("SEARCH_ENDPOINT")
        self.search_api_key = os.getenv("SEARCH_API_KEY")

        self.index_name = os.getenv("INDEX_NAME")
        self.vec_profile_name = os.getenv("VECTOR_SEARCH_PROFILE_NAME")
        self.vec_alg_conf_name = os.getenv("VECTOR_SEARCH_ALGORITHM_CONFIGURATION_NAME")
        self.sem_conf_name = os.getenv("SEMANTIC_CONFIGURATION_NAME")

        self.title_field = title_field
        self.use_vector = use_vector

    def get_search_client(self):
        search_client = SearchClient(
            endpoint=self.search_endpoint,
            credential=AzureKeyCredential(self.search_api_key),
            index_name=self.index_name
        )
        return search_client

    def get_search_index_client(self):
        search_index_client = SearchIndexClient(
            endpoint=self.search_endpoint, 
            credential=AzureKeyCredential(self.search_api_key),
            index_name=self.index_name
        )
        return search_index_client
    
    def index_exists(self, client: SearchIndexClient = None):
        if not client:
            client = self.get_search_index_client() # add try-except for closing

        indexes = client.list_index_names()
        return self.index_name in indexes

    def delete_index_if_exists(self):
        client = self.get_search_index_client()

        try:
            if self.index_exists():
                print(f"Index '{self.index_name}' exists.")
                client.delete_index(self.index_name)
                print(f"Index '{self.index_name}' deleted.")
            else:
                print(f"Index '{self.index_name}' does not exist.")
        except Exception as e:
            print(f"Error deleting index: {e}")

        client.close() # add try-except

    def create_search_index(self):

        # Define the index fields
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
            SearchableField(name="page", type=SearchFieldDataType.Int32),
            SearchableField(name="source", type=SearchFieldDataType.String),
            SearchableField(name="url", type=SearchFieldDataType.String),
        ]

        if self.use_vector: # otherwise use keyword-search only

            # Add vector field
            fields.append(
                SearchField(name="vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=1536, 
                    vector_search_profile_name=self.vec_profile_name,
                    )
                )

            # Set the vector search configuration  
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name=self.vec_alg_conf_name
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name=self.vec_profile_name,
                        algorithm_configuration_name=self.vec_alg_conf_name,
                    )
                ]
            )

        # Set the semantic configuration 
        semantic_config = SemanticConfiguration(
            name=self.sem_conf_name,
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="header"),
                content_fields=[SemanticField(field_name="raw_content")],
                keywords_fields=[SemanticField(field_name="page"), SemanticField(field_name="source")],
            )
        )

        semantic_search = SemanticSearch(configurations=[semantic_config])
        
        search_index = SearchIndex(name=self.index_name, fields=fields, vector_search=vector_search if self.use_vector else None, semantic_search=semantic_search)
        
        client = self.get_search_index_client()
        result = client.create_or_update_index(search_index) # add try-except
        print(f'Index "{result.name}" created')

        client.close() # add try-except

    def upload_to_index(self, data): # add overwrite
        search_client = self.get_search_client()

        for chunk in data:
            local_file_name = chunk[self.title_field]

            try:
                result = search_client.upload_documents(documents=json.dumps(chunk))
                print(f"\nUpload of {local_file_name} succeeded: { result[0].succeeded }")
                # print("Upload of '{}'.. succeeded? {}".format(local_file_name, result[0].succeeded))
            except Exception as e:
                print(f"Could not upload + {local_file_name} ({e})")
            
        search_client.close() # add try-except
