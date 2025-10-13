from utils.azure_utils import *

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

from pydantic import BaseModel, Field
from typing import Dict, List


SYSTEM_MESSAGE_TEMPLATE = (
"""\
You're an expert of Contoso corporation that can answer all user questions.
Given a user question and a tagged list of retrieved sources ordered by relevance, provide a comprehensive response, grounded on the provided sources, that answer the user query.
In your answer, always ground your statements by citing the sources using their tag (doc id within squared brackets) - i.e. [doc_id].
Before answering provide also some sentences explaining how you interpreted the intent of the user query.
In general, allow user queries that ask to manipulate information and to reason about sources relationships for personal purposes but ensure the topics are related to retrieved information.
If the topic of the user query is not related to the retrieved sources, just state: 'The requested information is not available in the retrieved data. Please try another query or topic.'
When the user query is related to the topics in the retrieved sources the argument 'related' is positive, otherwise negative.

RETRIEVED SOURCES:

{retrieved_sources}

You must cite sources by their tag [doc_i] in order to ground your statements.
NOT cite by their title or else.
"""
)

class Response(BaseModel):
    intent: List[str] = Field("One or more sentences that represent how you interpreted the user query. They can be questions, queries, etc.")
    related: bool = Field("True if the user query is related to the retrieved sources else False")
    answer: str = Field("Comprehensive but compact answer to the user query grounded on the provided sources cited with their tags, i.e. [doc_i]")


def get_response(query: str, messages: List[Dict], search_type: str = "vector_text"):
    """
    Returns a response from the OpenAI client.
    """

    assert search_type in search_types

    llm = get_openai_client()
    llm = llm.with_structured_output(Response)

    use_semantic_search = 'semantic' in search_type

    search_results = retrieve(query, semantic=use_semantic_search)

    sources_formatted = format_sources(search_results)
    # with open("all_sources.txt","r") as fin:
    #     sources_formatted = fin.read()
        # print(sources_formatted)

    sys_msg = PromptTemplate.from_template(SYSTEM_MESSAGE_TEMPLATE).format(retrieved_sources=sources_formatted)
    sys_msg = sys_msg.strip("\n")

    # print(sys_msg)

    new_messages = ChatPromptTemplate(
            [
                ("system", "{sys_msg}"),
                ("human", "{query}"),
            ]
    ).format_messages(sys_msg=sys_msg, query=query)

    # print(new_messages)

    chat = messages + new_messages

    # print(chat)

    response = llm.invoke(chat)

    return response.answer, response.intent, response.related, sources_formatted


def main(search_type: str):
    history = []
    
    while True:
        print("\n❌ Type 'exit' to quit.\n\n\n")

        user_query = input("🔍 Question: ")

        if user_query.lower() == "exit":
            break

        # user_words = re.split(r'"|\'', user_query)
        # image_url = None
        # for x in user_words:
        #     if x.endswith((".png", ".jpeg", ".jpg")):
        #         local_image_name = x
        #         image_path = os.path.join("Contoso Corp.", "images", local_image_name)
        #         image_url = local_image_to_data_url(image_path)        
        
        # if image_url:
        #     msg = {"role": "user", "content": [           
        #                 { 
        #                     "type": "text", 
        #                     "text": "Describe this picture:" 
        #                 },
        #                 { 
        #                     "type": "image_url",
        #                     "image_url": {
        #                         "url": f"{image_url}"
        #                     }
        #                 }
        #             ]}
        # else:
        #     msg = {"role": "user", "content": user_query}

        # messages = history

        answer, intent, related, sources, error = None, None, None, None, None
        
        spinner_thread = SpinnerThread() # spinner for entertainment while waiting llm generation
        spinner_thread.start()

        try:
            answer, intent, related, sources = get_response(query=user_query, messages=history, search_type=search_type) # LLM response
        except Exception as e:
            error = f"\n⛔ {e}\n"

        spinner_thread.stop()
        spinner_thread.join()

        if error:
            print(error)
        
        if answer:
            print("\n🤖 Answer: ", answer)

            if related:
                # user query is related to the topics in the retrieved sources
                
                if intent:
                    print(f"\n🎯 Intent: {intent}")

                if sources:
                    print(f"\n📚 Sources:\n\n{sources}")

                # update messages history
                history.append(HumanMessage(content=user_query))
                history.append(AIMessage(content=answer))
        
        input("\n↩️  Press a key..")

        print ("\033[A\033[2K\033[A") # ANSI ESCAPE CODES
        print("--------------------------------------------------")



if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", 
                        "--search-type", 
                        type=str, 
                        default="vector_text", 
                        choices=search_types, 
                        help="type of search (default: 'vector_text')"
                        )

    args = parser.parse_args()
    search_type = args.search_type

    main(search_type=search_type)
