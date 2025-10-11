import os
from dotenv import load_dotenv
load_dotenv(".env", override=True)

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient

from langchain_openai import AzureChatOpenAI

import base64
from mimetypes import guess_type

import sys
import threading
import time


openai_endpoint = os.getenv("OPENAI_ENDPOINT")
openai_api_version = "2024-12-01-preview"

chat_api_key = os.getenv("CHAT_API_KEY")
chat_deployment = os.getenv("CHAT_DEPLOYMENT")

document_endpoint = os.getenv("DOCUMENT_ENDPOINT")
document_api_key = os.getenv("DOCUMENT_API_KEY")


def get_openai_client():
    """
    Returns an instance of the AzureChatOpenAI client.
    """
    
    return AzureChatOpenAI(
            azure_endpoint=openai_endpoint,
            api_version=openai_api_version,
            api_key=chat_api_key, 
            azure_deployment=chat_deployment, 
            temperature=0.1,
            max_completion_tokens=1000,
        )

def get_document_client():
    """
    Returns an instance of the DocumentIntelligenceClient client.
    """

    return DocumentIntelligenceClient(
            endpoint=document_endpoint, credential=AzureKeyCredential(document_api_key)
        )

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
