# PREBUILT READ

import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
import re
import tiktoken
from embedings import get_chunk_object
from index import create_search_index, upload_chunk_document, delete_index_if_exists
from storage import connect_to_storage
from indexer import get_indexer

import pickle
import json


def parse_paragraphs(analyze_result):
    table_offsets = []
    page_content = {}

    for paragraph in analyze_result.paragraphs:  
        for span in paragraph.spans:
            if span.offset not in table_offsets:
                for region in paragraph.bounding_regions:
                    page_number = region.page_number
                    if page_number not in page_content:
                        page_content[page_number] = []
                    page_content[page_number].append({
                        "content_text": paragraph.content
                    })
    return page_content, table_offsets

# def parse_paragraphs(analyze_result):
#     table_offsets = []
#     paragraphs_formatted = []

#     for paragraph in analyze_result.paragraphs:
#         for span in paragraph.spans:
#             if span.offset not in table_offsets:
#                 for region in paragraph.bounding_regions:
#                     role = paragraph.role.capitalize() if paragraph.role else "unnamed_paragraph"
#                     content = paragraph.content
#                     page = region.page_number
#                     paragraphs_formatted.append({
#                         "header":role,
#                         "format_content":"",
#                         "raw_content":content,
#                         "paragraph_number":"",
#                         "page":page,
#                         "source":"report.pdf"})
#     print(paragraphs_formatted)
#     return paragraphs_formatted

def format_paragraphs(raw_paragraphs):
    p1_pars_formatted = []
    last_header = None
    
    for page_num, raw_pars in raw_paragraphs.items():
        for i, par in enumerate(raw_pars):
            content = par["content_text"]
            par_format = None

            if i==0 and not par_format:
                par_format = {
                        "header":"title",
                        "content":content,
                        "page":page_num,
                        "file_name":"report.pdf"}
            elif len(content) > 50:
                par_format = {
                        "header":last_header if last_header and len(re.sub("\d+(?:\.\d+)", "",last_header))>5 else f"paragraph_{i+1}",
                        "content":content,
                        "page":page_num}
            else:
                last_header = content
            
            if par_format:
                p1_pars_formatted.append(par_format)

    return p1_pars_formatted

def analyze_read():
    # key = os.getenv("DOCUMENTINTELLIGENCE_API_KEY")
    # endpoint = os.getenv("DOCUMENTINTELLIGENCE_ENDPOINT")

    # document_intelligence_client = DocumentIntelligenceClient(
    #     endpoint=endpoint, credential=AzureKeyCredential(key)
    # )

    # file = open('sample.pdf', "rb")

    # poller = document_intelligence_client.begin_analyze_document(
    #     "prebuilt-layout", body=file
    # )

    # result = poller.result()

    # with open("sample_result.pkl", "wb") as fout:
    #     pickle.dump(result, fout)

    ### Bypass DocumentIntelligence call and load the pickle to avoid exceeding the Azure free subscription limit of 500 pages per month

    with open("sample_result.pkl", 'rb') as fin:
        result = pickle.load(fin)

    parags = parse_paragraphs(result)[0]
    p1_parags = format_paragraphs(parags)


    # for table_idx, table in enumerate(result.tables):
    #     print(
    #         "Table # {} has {} rows and {} columns, {}".format(
    #         table_idx, table.row_count, table.column_count, table.footnotes
    #         )
    #     )
            
    #     for cell in table.cells:
    #         print(
    #             "...Cell[{}][{}] has content '{}'".format(
    #             cell.row_index,
    #             cell.column_index,
    #             cell.content.encode("utf-8"),
    #             )
    #         )

    tokenizer = tiktoken.get_encoding(encoding_name="cl100k_base")
    tokens = [len(tokenizer.encode(p["content"], disallowed_special=()))>1000 for p in p1_parags]
    assert not any(tokens)

    p1_parags = [get_chunk_object(x) for x in p1_parags]

    search_index_name = "es1sea"

    print("Deleting search index if it exists...")
    delete_index_if_exists(search_index_name)

    print("Creating new search index...")
    create_search_index(search_index_name)
    print(f"Search index '{search_index_name}' created.")

    connect_to_storage()

    from azure.identity import DefaultAzureCredential
    from azure.core.credentials import AzureKeyCredential
    from azure.storage.blob import BlobServiceClient

    account_url = "https://es1sto.blob.core.windows.net"
    
    blob_service_client = BlobServiceClient(account_url, credential=os.getenv("STORAGE_API_KEY"))
    # for par in p1_parags:
    #     local_file_name = par["header"]
    #     blob_client = blob_service_client.get_blob_client(container="es1cont", blob=local_file_name)

    #     print("\nUploading to Azure Storage as blob:\n\t" + local_file_name)

    #     # Upload the created file
    #     blob_client.upload_blob(json.dumps(par))
    
    get_indexer()

    # print("Uploading chunks to search index...")
    # for par in p1_parags:
    #     upload_chunk_document(par, search_index_name)




if __name__ == "__main__":
    from azure.core.exceptions import HttpResponseError
    from dotenv import find_dotenv, load_dotenv

    try:
        load_dotenv(find_dotenv())
        analyze_read()
    except HttpResponseError as error:
        if error.error is not None:
            print(f"Received service error: {error.error}")
            raise
        if "Invalid request".casefold() in error.message.casefold():
            print(f"Invalid request: {error}")
        raise