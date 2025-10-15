from utils.azure_utils import get_response, SEARCH_TYPES

from pydantic import BaseModel, Field
from typing import List

from utils.utils import SpinnerThread


SYSTEM_MESSAGE_TEMPLATE = (
"""
You're an expert of Contoso corporation that can answer all user questions.
Given a user question and a tagged list of retrieved sources ordered by relevance, provide a comprehensive response, grounded on the provided sources, that answer the user query.
In your answer, always ground your statements by citing the sources using their tags; source tags are within square brackets, e.g.: [doc_i]
Before answering provide also some sentences explaining how you interpreted the intent of the user query.
In general, allow also user queries that ask to manipulate information and to reason about sources relationships for personal purposes but ensure the topics in the queries are related to the retrieved information.
If the topic of the user query is not related to the retrieved sources, just state: 'The requested information is not available in the retrieved data. Please try another query or topic.'
When the user query is related to the topics in the retrieved sources the argument 'related' is positive, otherwise negative.

RETRIEVED SOURCES:

{sources}

You MUST cite sources by their tag [doc_i] in order to ground your statements.
Do NOT cite sources by their title or other fields.
"""
)

class Response(BaseModel):
    related: bool = Field("True if the user query is related to the retrieved sources else False")
    intent: List[str] = Field("One or more sentences that represent how you interpreted the user query. They can be questions, queries, reasoning, etc.")
    answer: str = Field("Comprehensive but compact answer to the user query grounded on the provided sources cited with their tags, i.e. [doc_i]")


def main(search_type: str, use_history: bool = True):
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

        messages = history if use_history else []

        answer, intent, related, sources, error = None, None, None, None, None
        
        spinner_thread = SpinnerThread() # spinner for entertainment while waiting the response
        spinner_thread.start()

        try:
            response, sources, _ = get_response(query=user_query, 
                                             sys_template=SYSTEM_MESSAGE_TEMPLATE, 
                                             messages=messages, 
                                             search_type=search_type, 
                                             openai_kwargs={"temperature":0.1, "max_tokens":None},
                                             output_schema=Response) # LLM response + retrieved sources

            # extract results from output schema
            related, intent, answer = response.related, response.intent, response.answer
        except Exception as e:
            error = f"\n⛔ {e}\n"

        spinner_thread.stop()
        spinner_thread.join()

        if error:
            print(error)
        
        if answer:
            print("\n🤖 Answer: ", answer)

            if related:
                # user query is related to the topics of the retrieved sources
                
                if intent:
                    print(f"\n🎯 Intent: {intent}")

                if sources:
                    print(f"\n📚 Sources:\n\n{sources}")

                # update messages history
                history.append({"role": "user", "content": f"{user_query}"})
                history.append({"role": "assistant", "content": f"{answer}"})
        
        input("\n↩️  Press a key..")

        print ("\033[A\033[2K\033[A") # ANSI ESCAPE CODES
        print("--------------------------------------------------")



if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-nh",
                        "--no-history",
                        action="store_true",
                        help="prevent from passing chat history to llm"
    )
    parser.add_argument("-s", 
                        "--search-type", 
                        type=str, 
                        default="vector_text", 
                        choices=SEARCH_TYPES, 
                        help="type of search (default: 'vector_text')"
    )

    args = parser.parse_args()

    use_history = not args.no_history
    search_type = args.search_type

    main(use_history=use_history, search_type=search_type)
