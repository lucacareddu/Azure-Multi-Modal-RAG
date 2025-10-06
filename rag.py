import os
from dotenv import load_dotenv
load_dotenv(".env", override=True)

from openai import AzureOpenAI

from rag_utils import SpinnerThread


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
        azure_endpoint=chat_endpoint,
        api_key=chat_api_key,
    )

def get_response(messages):
    """
    Returns a response from the OpenAI client.
    """

    client = get_openai_client()

    data_source = {
                    "type": "azure_search",
                    "parameters": {
                        "endpoint": search_endpoint,
                        "index_name": index_name,
                        "authentication": {
                            "type": "api_key",
                            "key": str(search_api_key),
                        },
                        "query_type": "vector",
                        "embedding_dependency": {
                            "type": "deployment_name",
                            "deployment_name": str(embedding_deployment),
                        },
                        "semantic_configuration": str(semant_config_name),
                        "in_scope": True,
                        "top_n_documents": 3,
                        "strictness": 3,
                        "fields_mapping": {
                            "content_fields_separator": "\\n",
                            "title_field": "header",
                            "content_fields": [
                                "raw_content"
                            ],
                            "filepath_field": "source",
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
        model=chat_deployment
    )

    return response


def main():
    history = [{
                "role": "system",
                "content": "You are an expert report analyzer that provide relevant information to users querying it",
        }]

    
    while True:
        print("\n❌ Type 'exit' to quit.\n\n\n")

        user_input = input("🔍 Question: ")

        if user_input.lower() == "exit":
            break
        
        msg = {"role": "user", "content": user_input}
        messages = history + [msg]
        response = None
        
        
        spinner_thread = SpinnerThread() # spinner for entertainment while waiting llm generation
        spinner_thread.start()

        try:
            response = get_response(messages) # LLM response
            error = None
        except Exception as e:
            error = f"\n⛔ {e}\n"

        spinner_thread.stop()
        spinner_thread.join()

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
                    print(f"Title: {citation['title']}")
                    print(f"Content: {citation['content']}")

            history.append(msg)
            history.append({"role": "assistant", "content": answer})
        
        input("\n↩️  Press a key..")

        print ("\033[A\033[2K\033[A") # ANSI ESCAPE CODES
        print("--------------------------------------------------")



if __name__=="__main__":
    main()
