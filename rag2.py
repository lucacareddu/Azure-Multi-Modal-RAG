# Set up the query for generating responses
from azure.identity import DefaultAzureCredential
from azure.identity import get_bearer_token_provider
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI

import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

AZURE_OPENAI_ACCOUNT = os.getenv("OPENAI_ENDPOINT")
api_key = os.getenv("OPENAI_API_KEY")
deployment = os.getenv("OPENAI_DEPLOYMENT")
AZURE_SEARCH_SERVICE = os.getenv("SEARCH_ENDPOINT")

endpoint_ai_search = os.getenv("SEARCH_ENDPOINT")
api_key_ai_search = os.getenv("SEARCH_API_KEY")
search_index_name = os.getenv("INDEX_NAME")
deployment = os.getenv("OPENAI_DEPLOYMENT")
api_version = "2024-12-01-preview"


# credential = DefaultAzureCredential()
# token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
openai_client = AzureOpenAI(
    api_version=api_version,#"2024-06-01",
    azure_endpoint=AZURE_OPENAI_ACCOUNT,
    api_key=api_key
    # azure_ad_token_provider=token_provider
)

search_client = SearchClient(
    endpoint=AZURE_SEARCH_SERVICE,
    index_name=search_index_name,
    credential=AzureKeyCredential(api_key_ai_search)
)

# This prompt provides instructions to the model
GROUNDED_PROMPT="""
You are an expert report analyzer that proxies relevant information in reports to users querying for them.
Query: {query}
Sources:\n{sources}
"""

# Query is the question being asked. It's sent to the search engine and the chat model
query="Disgenet"

# Search results are created by the search client
# Search results are composed of the top 5 results and the fields selected from the search index
# Search results include the top 5 matches to your query
search_results = search_client.search(
    search_text=query,
    top=5,
    select="header,raw_content,paragraph,page"
)
sources_formatted = "\n\n".join([f'Title: {document["header"]}\nCorpus: {document["raw_content"]}\nParagraph: {document["paragraph"]}' for document in search_results])
# print(sources_formatted)

# Send the search results and the query to the LLM to generate a response based on the prompt.
response = openai_client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": GROUNDED_PROMPT.format(query=query, sources=sources_formatted)
        }
    ],
    model=deployment
)

# Here is the response from the chat model.
print(response.choices[0].message.content)

citations = response.choices[0].message
print(citations)
# if citations:
#     print("Citations: ", response.choices[0].message.context["citations"])
#     for citation in citations:
#         print(f" - {citation['title']}: {citation['filepath']}")