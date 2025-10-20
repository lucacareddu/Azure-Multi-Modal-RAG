import os
import shutil
from PyPDF2 import PdfReader, PdfWriter
import json


pdf_root = "Microsoft AI study"
pdf_names = ["2024-State-of-AI-Change-Readiness-eBook.pdf"]
pdf_paths = [os.path.join(pdf_root, pdf_name) for pdf_name in pdf_names]
pdf_urls = ["https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/2024-State-of-AI-Change-Readiness-eBook.pdf"]

for path in pdf_paths:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"'{path}' does not exist.")

split_dir_name = "splits"
split_dir_path = os.path.join(pdf_root, split_dir_name)
shutil.rmtree(split_dir_path) # remove if it exists
os.makedirs(split_dir_path)

microsoftAIStudy_pages = [list(range(2,23))]

func = lambda name,pages: [name.replace(".pdf", f"__{x[0]}-{x[-1]}.pdf") for x in pages]

split_map = {
            "name": pdf_names,
            "path": pdf_paths,
            "split_name": [func(pdf_names[0], microsoftAIStudy_pages),],
            "pages": [microsoftAIStudy_pages,],
            "url": pdf_urls
            }

num_pdfs = len(pdf_names)

for i in range(num_pdfs):
    name = split_map["name"][i]
    path = split_map["path"][i]
    url = split_map["url"][i]

    reader = PdfReader(open(path, "rb"))

    for split_name, pages in zip(split_map["split_name"][i], split_map["pages"][i]):
        writer = PdfWriter()
        for pg in pages:
            writer.add_page(reader.pages[pg-1])

        new_path_pdf = os.path.join(split_dir_path,split_name)
        new_path_json = os.path.join(split_dir_path,split_name.replace(".pdf", ".json"))
        
        with open(new_path_pdf, "wb") as output_pdf:
            writer.write(output_pdf)
            writer.close()
        
        with open(new_path_json, "w") as output_json:
            metadata = json.dumps({
                "file_name": name,
                "split_offset": pages[0],
                "url": url
            })
            output_json.write(metadata)
        