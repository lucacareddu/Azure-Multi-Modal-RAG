import os
import glob
import uuid
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

import json
import pickle


class DocumentProcessor():
    """
        Scan and format information from pdfs
    """

    def __init__(self, root_path: str, splits_dir: str = "splits"):
        splits_path = os.path.join(root_path, splits_dir)
        assert os.path.isdir(splits_path)

        self.root_path = root_path
        self.splits_path = splits_path

        self.document_endpoint = os.getenv("DOCUMENT_ENDPOINT")
        self.document_api_key = os.getenv("DOCUMENT_API_KEY")

    def get_document_client(self):
        document_intelligence_client = DocumentIntelligenceClient(
            endpoint=self.document_endpoint, credential=AzureKeyCredential(self.document_api_key)
        )
        return document_intelligence_client

    def create_from_pkl(self, path_to_file):
        with open(path_to_file, 'rb') as fin:
            result = pickle.load(fin)

        return result
    
    def create_from_layout(self, save_to_file="results.pkl"):
        document_intelligence_client = self.get_document_client()

        pdf_files = glob.glob("*.pdf", root_dir=self.splits_path)
        json_files = glob.glob("*.json", root_dir=self.splits_path)

        pdf_files.sort()
        json_files.sort()

        assert len(pdf_files) == len(json_files)

        results = []

        for pdf_name, json_name in zip(pdf_files, json_files):
            pdf_path = os.path.join(self.splits_path, pdf_name)
            json_path = os.path.join(self.splits_path, json_name)

            pdf_file = open(pdf_path, "rb")

            poller = document_intelligence_client.begin_analyze_document(
                "prebuilt-layout", body=pdf_file
            )

            result = poller.result()

            with open(json_path, "r") as json_file:
                json_file = json.load(json_file)

            results.append({
                **json_file,
                "result": result
                })

        document_intelligence_client.close() # add try-except

        if save_to_file:
            with open(save_to_file, "wb") as fout:
                pickle.dump(results, fout)         

        return results
    
    def get_tables(self, analyze_result, visualize=False):
        tables, spans = [], []

        for table_idx, table in enumerate(analyze_result.tables):
            table_min_offset = min([span["offset"] for span in table.spans])
            table_max_length = max([span["length"] for span in table.spans])
            table_span = {"offset": table_min_offset, "length": table_max_length}

            spans.append(table_span)

            header = "Table{} with {} rows and {} columns.".format(
                (f" ({table.footnotes[-1].content})" if table.footnotes else ""), table.row_count, table.column_count
                )
            
            if visualize:
                print(header)

            content = ""
                
            for cell in table.cells:
                cell_to_string = "Cell[{}][{}]: {}".format(
                            cell.row_index,
                            cell.column_index,
                            cell.content,#.encode("utf-8"),
                            )

                if visualize:
                    print(cell_to_string)
                
                content += " " + cell_to_string
            
            tables.append({"header":header, "content":content})

        return tables, spans

    def format_paragraphs(self, analyze_result, add_id: bool = True):
        file_name = analyze_result["file_name"]
        split_offset = analyze_result["split_offset"]
        result = analyze_result["result"]
        url = analyze_result["url"]

        tables, tables_spans = self.get_tables(result)

        formatted_paragraphs = []

        header = None

        for paragraph in result.paragraphs:
            for span in paragraph.spans:
                for region in paragraph.bounding_regions:
                    
                    # check if paragraph is/contains a table
                    table = None
                    for t, s in zip(tables, tables_spans):
                        if s["offset"] <= span["offset"] <= (s["offset"]+s["length"]):
                            table = t

                    # if table, check if not already in results
                    if table and (not formatted_paragraphs or table["content"] != formatted_paragraphs[-1]["raw_content"]):
                        table_header = table["header"]
                        content = table["content"]
                        page_num = int(region.page_number) + int(split_offset) - 1

                        form_par = {"id": str(uuid.uuid4())} if add_id else {}

                        formatted_paragraphs.append({
                            **form_par,
                            "header":table_header,
                            "raw_content":content,
                            "format_content":content,
                            "page":page_num,
                            "source":file_name,
                            "url":url})
                                
                    role = paragraph.role

                    if not table and role:            
                        if role in [ParagraphRole.TITLE, ParagraphRole.SECTION_HEADING]:
                            header = paragraph.content
                        
                    elif not table and not role:
                        content = paragraph.content
                        page_num = int(region.page_number) + int(split_offset) - 1

                        # normalize text
                        format_content = paragraph.content.strip().lower()
                        format_content = re.sub(r'\d+','',format_content)
                        format_content = re.sub(r'[^\w\s]','',format_content)

                        lemmmatizer = WordNetLemmatizer()
                        words = [lemmmatizer.lemmatize(word) for word in format_content.split() if word not in set(stopwords.words('english'))]
                        format_content = ' '.join(words)
                        

                        if formatted_paragraphs and header == formatted_paragraphs[-1]["header"]:
                            # append to existing paragraph
                            formatted_paragraphs[-1]["raw_content"] += "\n" + content
                            formatted_paragraphs[-1]["format_content"] += " " + format_content
                        else:
                            form_par = {"id": str(uuid.uuid4())} if add_id else {}
                            
                            formatted_paragraphs.append({
                                **form_par,
                                "header":header,
                                "raw_content":content,
                                "format_content":format_content,
                                "page":page_num,
                                "source":file_name,
                                "url":url})

        return formatted_paragraphs
    
    def visualize_result(self, result):
        for i, res in enumerate(result):
            print(f"Split #{i+1}\n")
            for par in res:
                for k,v in par.items():
                    print(k)
                    print(v)
                    print()
                input("")
                print("\n\n\n\n")

    def flatten_result(self, result):
        flattened_results = []

        for res in result:
            flattened_results.extend(res)

        return flattened_results
    
    def format_result(self, analyze_result, flatten=False):
        formatted_results = []

        for res in analyze_result:
            formatted_results.append(self.format_paragraphs(res))

        if flatten:
            formatted_results = self.flatten_result(formatted_results)

        return formatted_results
    