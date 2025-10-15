import re
from typing import List

import base64
from mimetypes import guess_type

import sys
import threading
import time


def tags_to_headers(tags: List, sources: List, header_field: str = "header"):
    indices = set(int(re.findall(r"doc_(\d+)", tag)[0]) for tag in tags)
    sources_headers = [sources[i-1][header_field] for i in indices]
    return sources_headers


def format_sources(sources, reference=None):
    if reference:
        sources_map = [doc["header"] for doc in reference]

    sources_formatted = ""
    sources_list = []

    for idx, doc in enumerate(sources):
        idx = idx if not reference else sources_map.index(doc['header'])

        source = "\n".join([
                            f"[doc_{idx+1}]",
                            f"TITLE: {doc['header']}",
                            f"CONTENT: {doc['raw_content']}",
                            f"SOURCE: {doc['source']}",
                            f"PAGE: {doc['page']}",
                            f"URL: {doc['url']}"
                            ])
        
        sources_formatted += source
        sources_formatted += "\n\n"

        sources_list.append(doc)
    
    sources_formatted = sources_formatted.strip()

    return sources_formatted.strip(), sources_list


### FROM "https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/gpt-with-vision?tabs=python"
# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

# Example usage
# image_path = '<path_to_image>'
# data_url = local_image_to_data_url(image_path)
# print("Data URL:", data_url)


class SpinnerThread(threading.Thread):

    def __init__(self):
        super().__init__(target=self._spin)
        self._stopevent = threading.Event()

    def stop(self):
        self._stopevent.set()

    def _spin(self):
        while not self._stopevent.is_set():
            for t in '|/-\\':
                sys.stdout.write(t)
                sys.stdout.flush()
                time.sleep(0.1)
                sys.stdout.write('\b')

        sys.stdout.write("\033[2K") # ANSI ESCAPE CODES to avoid printing over other main thread print
