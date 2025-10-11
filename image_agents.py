from langgraph.graph import StateGraph, START, END
from langchain.schema import HumanMessage

from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from utils import get_openai_client, get_document_client, local_image_to_data_url


class ResultOutput(BaseModel):
    title: str = Field("A title for the image that summarizes in one line its content")
    description: str = Field("A comprehensive and detailed but compact (not too long) description of the image")

class State(TypedDict):
    image: str
    description: str
    extracted_text: str
    result: ResultOutput

def description_agent(state: State) -> State:
    llm = get_openai_client()
    image_url = local_image_to_data_url(state["image"])

    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe the image"},
            {
                "type": "image_url",
                "image_url": {"url": image_url}
            },
        ]
    )

    ai_msg = llm.invoke([message])

    description = ai_msg.content
    return {"description": description}

def text_extractor_agent(state: State) -> State:
    doc = get_document_client()

    file = open(state["image"], "rb")

    poller = doc.begin_analyze_document(
        "prebuilt-layout", body=file
    )

    extracted_text = poller.result()["content"]
    return {"extracted_text": extracted_text}

def result_agent(state: State) -> State:
    llm = get_openai_client()
    llm = llm.with_structured_output(ResultOutput)

    prompt = f"""
            Given a general description and the exact extracted text contained in an image, produce your comprehensive and compact image description.
            The description must report relevant textual information from the extracted text.
            Report all percentages and numbers.
            Choose also a title for the image that summarizes in one line its main content.

            GENERAL DESCRIPTION: {state['description']}

            EXTRACTED TEXT: {state['extracted_text']}
            """
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
        ]
    )

    result = llm.invoke([message])

    return {"result": result}

def compile_graph():
    graph = StateGraph(State)

    graph.add_node("describe", description_agent)
    graph.add_node("extract", text_extractor_agent)
    graph.add_node("report", result_agent)

    graph.add_edge(START, "describe")
    graph.add_edge(START, "extract")
    graph.add_edge(["describe", "extract"], "report")

    graph.add_edge("report", END)

    workflow = graph.compile()

    return workflow

def draw_mermaid():
    app = compile_graph()

    png_image = app.get_graph().draw_mermaid_png()

    with open("graph.png", "wb") as f:
        f.write(png_image)



### USAGE 
       
# app = compile_graph()
# res = app.invoke({"image": "Contoso Corp./images/contoso_organization.png"})
# print(res["result"])
