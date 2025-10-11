import os
from dotenv import load_dotenv
load_dotenv(".env", override=True)

from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

import re
from rag_utils import SpinnerThread, local_image_to_data_url

from typing import Dict, List


chat_endpoint = os.getenv("CHAT_ENDPOINT")
chat_api_key = os.getenv("CHAT_API_KEY")
chat_deployment = os.getenv("CHAT_DEPLOYMENT")
api_version = "2024-12-01-preview"

embedding_deployment = os.getenv("EMBEDDING_DEPLOYMENT")

search_endpoint = os.getenv("SEARCH_ENDPOINT")
search_api_key = os.getenv("SEARCH_API_KEY")
index_name = os.getenv("INDEX_NAME")
semant_config_name = os.getenv("SEMANTIC_CONFIGURATION_NAME")


def get_openai_client():
    """
    Returns an instance of the AzureOpenAI client.
    """
    
    return AzureOpenAI(
        api_version=api_version,
        azure_endpoint="https://es1open.openai.azure.com",
        api_key=chat_api_key,
    )

def get_search_client():
    search_client = SearchClient(
        endpoint=search_endpoint,
        credential=AzureKeyCredential(search_api_key),
        index_name=index_name
    )
    return search_client

def get_embedding(text):
    client = get_openai_client()
    get_embeddings_response = client.embeddings.create(model=str(embedding_deployment), input=str(text))
    return get_embeddings_response.data[0].embedding

def retrieve(search_query: str):
    client = get_search_client()
    search_vector = get_embedding(search_query)
    result = client.search(search_query, top=3, vector_queries=[
        VectorizedQuery(vector=search_vector, k_nearest_neighbors=50, fields="vector")],
        query_type="semantic", semantic_configuration_name=semant_config_name)
    return result

def get_response(messages: List[Dict], search_type: str = "vector_semantic_hybrid"):
    """
    Returns a response from the OpenAI client.
    """

    assert search_type in ['simple', 'semantic', 'vector', 'vector_simple_hybrid', 'vector_semantic_hybrid']

    client = get_openai_client()
    
    if "vector" in search_type:
        data_source["parameters"]["embedding_dependency"] = {
                                                "type": "deployment_name",
                                                "deployment_name": str(embedding_deployment),
                                            }
        data_source["parameters"]["fields_mapping"]["vector_fields"] = ["vector"]
    
    response = client.chat.completions.create(
        messages=messages,
        extra_body={
                    "data_sources": [data_source]
                },
        stream=False,
        temperature=0.1,
        top_p=1.0,
        model=chat_deployment
    )

    return response


def main(search_type: str):
    history = [{
                "role": "system",
                "content": "You are an expert report analyzer that provide relevant information to users querying it",
        }]

    
    while True:
        print("\n❌ Type 'exit' to quit.\n\n\n")

        user_input = input("🔍 Question: ")

        if user_input.lower() == "exit":
            break

        for doc in retrieve(user_input):
            print(doc.keys())
        
        user_words = re.split(r'"|\'', user_input)
        image_url = None
        for x in user_words:
            if x.endswith((".png", ".jpeg", ".jpg")):
                local_image_name = x
                image_path = os.path.join("Contoso Corp.", "images", local_image_name)
                image_url = local_image_to_data_url(image_path)        
        
        if image_url:
            msg = {"role": "user", "content": [           
                        { 
                            "type": "text", 
                            "text": "Describe this picture:" 
                        },
                        { 
                            "type": "image_url",
                            "image_url": {
                                "url": f"{image_url}"
                            }
                        }
                    ]}
        else:
            msg = {"role": "user", "content": user_input}

        messages = history + [msg]
        response = None
        
        # spinner_thread = SpinnerThread() # spinner for entertainment while waiting llm generation
        # spinner_thread.start()

        try:
            response = get_response(messages, search_type=search_type) # LLM response
            error = None
        except Exception as e:
            error = f"\n⛔ {e}\n"

        # spinner_thread.stop()
        # spinner_thread.join()

        if error:
            print(error)
        
        if response:
            answer = response.choices[0].message.content
            print("\n🤖 Answer: ", answer)

            intent = response.choices[0].message.context["intent"]
            citations = response.choices[0].message.context["citations"]
            if citations and intent != "[]":
                print(f"\n🎯 Intent: {intent}")
                print("\n📚 Citations:")

                for idx, citation in enumerate(citations):
                    print()
                    print(f"[doc{idx+1}]")
                    print(f"TITLE: {citation['title']}")
                    print(f"CONTENT: {citation['content']}")
                    print(f"SOURCE: {citation['filepath']}")
                    print(f"URL: {citation['url']}")

            history.append(msg)
            history.append({"role": "assistant", "content": answer})
        
        input("\n↩️  Press a key..")

        print ("\033[A\033[2K\033[A") # ANSI ESCAPE CODES
        print("--------------------------------------------------")



if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--search-type", type=str, default="vector_semantic_hybrid", choices=['simple', 'semantic', 'vector', 'vector_simple_hybrid', 'vector_semantic_hybrid'], help="type of search (default: 'vector_semantic_hybrid')")

    args = parser.parse_args()
    search_type = args.search_type

    main(search_type=search_type)
