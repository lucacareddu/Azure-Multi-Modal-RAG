import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import ParagraphRole

import re

import nltk
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import pickle


class DocumentProcessor():
    """
        Scan and format information from pdfs
    """

    def __init__(self):
        self.document_endpoint = os.getenv("DOCUMENT_ENDPOINT")
        self.document_api_key = os.getenv("DOCUMENT_API_KEY")

    def get_document_client(self):
        document_intelligence_client = DocumentIntelligenceClient(
            endpoint=self.endpoint, credential=AzureKeyCredential(self.key)
        )
        return document_intelligence_client

    def create_from_pkl(self, path_to_file):
        with open(path_to_file, 'rb') as fin:
            result = pickle.load(fin)

        return result
    
    def create_from_layout(self, path_to_file, save_to_file=None):
        document_intelligence_client = self.get_document_client()

        file = open(path_to_file, "rb")

        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-layout", body=file
        )

        result = poller.result()

        document_intelligence_client.close() # add try-except

        if save_to_file:
            with open(save_to_file, "wb") as fout:
                pickle.dump(result, fout)

        return result
    
    def format_tables(self, analyze_result, formatted_paragraphs):
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

    def format_paragraphs(self, analyze_result):
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
        
        # self.format_tables(analyze_result, formatted_paragraphs)

        return formatted_paragraphs
