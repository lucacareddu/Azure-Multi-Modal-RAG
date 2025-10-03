# import json
import os
from dotenv import load_dotenv

from openai import AzureOpenAI

load_dotenv(".env", override=True)

# api_key = os.getenv("AZURE_AI_KEY")
# endpoint_ai = os.getenv("AZURE_AI_ENDPOINT")
endpoint_ai_search = os.getenv("SEARCH_ENDPOINT")
api_key_ai_search = os.getenv("SEARCH_API_KEY")
search_index_name = os.getenv("INDEX_NAME")
deployment = os.getenv("OPENAI_DEPLOYMENT")
api_version = "2024-12-01-preview"

def get_openai_client():
    """
    Returns an instance of the AzureOpenAI client.
    """
    return AzureOpenAI(
        api_version=api_version,
        azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

def get_response(messages):
    """
    Returns a response from the OpenAI client.
    """
    client = get_openai_client()

    data_source = {
                        "type": "azure_search",
                        "parameters": {
                            "endpoint": endpoint_ai_search,
                            "index_name": search_index_name,
                            "authentication": {
                                "type": "api_key",
                                "key": str(api_key_ai_search),
                            },
                            "query_type": "vector",
                            "embedding_dependency": {
                            "type": "deployment_name",
                            "deployment_name": str(os.getenv("EMBEDDING_DEPLOYMENT")),
                        },
                        "semantic_configuration": "my-semantic-config",
                        "in_scope": True,
                        "top_n_documents": 3,
                        "strictness": 3,
                        # "role_information": "You are an AI assistant that helps people find information.",
                        "fields_mapping": {
                            "content_fields_separator": "\\n",
                            "content_fields": [
                            "raw_content"
                            ],
                            "filepath_field": "source",
                            "title_field": "header",
                            # "url_field": "url",
                            "vector_fields": [
                            "vector"
                            ]
                            }
                        }
                    }
    
    response = client.chat.completions.create(
        messages=messages,
        extra_body={
                    "data_sources": [data_source]
                },
        stream=False,
        temperature=0.1,
        top_p=1.0,
        model=deployment
    )
    return response


messages = [{
            "role": "system",
            "content": "You are an expert in the economic theory or Henry George. An american economist of the 19 century",
    }]


while True:
    user_input = input("User: what is your question? (type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    messages.append({"role": "user", "content": user_input})

    client = get_openai_client()

    response = get_response(messages)
    print(response.choices[0].message.content)
    citations = response.choices[0].message.context["citations"]
    if citations:
        print("Citations: ", response.choices[0].message.context["citations"])
        for citation in citations:
            print(f" - {citation['title']}: {citation['filepath']}")
    else:
        print("No citations found.")
    messages.append({"role": "assistant", "content": response.choices[0].message.content})
