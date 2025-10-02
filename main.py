# PREBUILT READ

import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import ParagraphRole
from embedings import get_chunk_object
from index import create_search_index, delete_index_if_exists
from storage import connect_to_storage
from indexer import get_indexer

import pickle
import re

# import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import tiktoken
import json


def analyze_layout(path_to_file, save_to_file=None):
    key = os.getenv("DOCUMENTINTELLIGENCE_API_KEY")
    endpoint = os.getenv("DOCUMENTINTELLIGENCE_ENDPOINT")

    document_intelligence_client = DocumentIntelligenceClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )

    file = open(path_to_file, "rb")

    poller = document_intelligence_client.begin_analyze_document(
        "prebuilt-layout", body=file
    )

    result = poller.result()

    if save_to_file:
        with open(save_to_file, "wb") as fout:
            pickle.dump(result, fout)

    return result

def upload_to_container(data, overwrite=True):
    from azure.storage.blob import BlobServiceClient

    storage_name = os.getenv("STORAGE_NAME")
    account_url = f"https://{storage_name}.blob.core.windows.net"
    api_key = os.getenv("STORAGE_API_KEY")

    blob_service_client = BlobServiceClient(account_url=account_url, credential=api_key)

    container_name = os.getenv("CONTAINER_NAME")
    
    for chunk in data:
        local_file_name = chunk["header"]
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=local_file_name)

        # try:
        #     blob_client.delete_blob()
        # except:
        #     print("Could not delete " + local_file_name)
        #     pass
        
        print("\nUploading to Azure Storage as blob:\n\t" + local_file_name)

        # Upload the created file
        blob_client.upload_blob(json.dumps(chunk), overwrite=overwrite)

def get_tables(analyze_result, formatted_paragraphs):
    for table_idx, table in enumerate(analyze_result.tables):
        print(
            "Table # {} ({}) has {} rows and {} columns, {}".format(
            table_idx, table.content, table.row_count, table.column_count, table.footnotes
            )
        )
            
        for cell in table.cells:
            print(
                "...Cell[{}][{}] has content '{}'".format(
                cell.row_index,
                cell.column_index,
                cell.content.encode("utf-8"),
                )
            )
    return

def format_paragraphs(analyze_result):
    table_offsets = []
    formatted_paragraphs = []

    header = None
    for paragraph in analyze_result.paragraphs:
        for span in paragraph.spans:
            paragraph
            if span.offset not in table_offsets:
                for region in paragraph.bounding_regions:
                    role = paragraph.role
                    if role == ParagraphRole.TITLE:
                        header = "title"
                    elif role == ParagraphRole.SECTION_HEADING:
                        header = paragraph.content
                        continue
                    if header and role != ParagraphRole.PAGE_NUMBER:
                        par_num = len(formatted_paragraphs)+1
                        header = header if header else f"paragraph_{len(formatted_paragraphs)+1}"
                        content = paragraph.content
                        page_num = int(region.page_number)

                        format_content = paragraph.content.strip().lower()
                        format_content = re.sub(r'\d+','',format_content)
                        format_content = re.sub(r'[^\w\s]','',format_content)

                        lemmmatizer = WordNetLemmatizer()
                        words = [lemmmatizer.lemmatize(word) for word in format_content.split() if word not in set(stopwords.words('english'))]
                        format_content = ' '.join(words)


                        formatted_paragraphs.append({
                            "header":header,
                            "raw_content":content,
                            "format_content":format_content,
                            "paragraph":par_num,
                            "page":page_num,
                            "source":"report.pdf"})
                        
                        header = None
    
    # get_tables(analyze_result, formatted_paragraphs)

    return formatted_paragraphs

def assess_tokens_number(paragraphs, max_tokens=1000):
    tokenizer = tiktoken.get_encoding(encoding_name="cl100k_base")
    tokens = [len(tokenizer.encode(p["format_content"], disallowed_special=())) > max_tokens for p in paragraphs]
    assert not any(tokens)


def main():
    # analyze_layout('sample.pdf', save_to_file="sample_result.pkl")

    ### Bypass DocumentIntelligence call and load the pickle to avoid exceeding the Azure free subscription limit of 500 pages per month

    with open("sample_result.pkl", 'rb') as fin:
        result = pickle.load(fin)

    parags = format_paragraphs(result)

    assess_tokens_number(parags) # Less than 1k

    parags = [get_chunk_object(x) for x in parags] # Add id and vector fields

    index_name = os.getenv("INDEX_NAME")

    print("Deleting search index if it exists...")
    delete_index_if_exists(index_name)

    print("Creating new search index...")
    create_search_index(index_name)
    print(f"Search index '{index_name}' created.")

    print("Creating new data source connection...")
    connect_to_storage()

    print("Loading data into the container...")
    upload_to_container(parags, overwrite=True)
    
    print("Creating new indexer...")
    get_indexer()



if __name__ == "__main__":
    from azure.core.exceptions import HttpResponseError
    from dotenv import find_dotenv, load_dotenv

    try:
        load_dotenv(find_dotenv())
        main()
    except HttpResponseError as error:
        if error.error is not None:
            print(f"Received service error: {error.error}")
            raise
        if "Invalid request".casefold() in error.message.casefold():
            print(f"Invalid request: {error}")
        raise