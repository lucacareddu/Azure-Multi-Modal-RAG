# PREBUILT READ

import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
import tiktoken
from embedings import get_chunk_object
from index import create_search_index, upload_chunk_document, delete_index_if_exists


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

def format_paragraphs(raw_paragraphs):
    p1_pars_formatted = []
    last_header = None
    
    for i, par in enumerate(raw_paragraphs):
        content = par["content_text"]
        par_format = None

        if i==0:
            par_format = {
                    "header":"title",
                    "content":content,
                    "page":"1",
                    "file_name":"report.pdf"}
        elif len(content) > 50:
            par_format = {
                    "header":last_header if last_header else f"paragraph_{i+1}",
                    "content":content,
                    "page":"1"}
        else:
            last_header = content
        
        if par_format:
            p1_pars_formatted.append(par_format)

    return p1_pars_formatted

def analyze_read():
    key = os.getenv("DOCUMENTINTELLIGENCE_API_KEY")
    endpoint = os.getenv("DOCUMENTINTELLIGENCE_ENDPOINT")

    document_intelligence_client = DocumentIntelligenceClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )

    file = open('sample.pdf', "rb")

    poller = document_intelligence_client.begin_analyze_document(
        "prebuilt-read", body=file
    )

    result = poller.result()

    p1_parags = parse_paragraphs(result)[0][1]
    p1_parags = format_paragraphs(p1_parags)

    tokenizer = tiktoken.get_encoding(encoding_name="cl100k_base")
    tokens = [len(tokenizer.encode(p["content"], disallowed_special=()))>1000 for p in p1_parags]
    assert not any(tokens)

    p1_parags = [get_chunk_object(x) for x in p1_parags]

    search_index_name = "report_index"

    print("Deleting search index if it exists...")
    delete_index_if_exists(search_index_name)

    print("Creating new search index...")
    create_search_index(search_index_name)
    print(f"Search index '{search_index_name}' created.")

    print("Uploading chunks to search index...")
    for par in p1_parags:
        upload_chunk_document(par, search_index_name)




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