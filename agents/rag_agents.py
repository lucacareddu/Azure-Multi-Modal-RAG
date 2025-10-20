from langgraph.graph import StateGraph, START, END

from typing_extensions import TypedDict, Dict, List
from pydantic import BaseModel, Field

from utils.azure_utils import get_chatopenai_client, get_response
from langchain_core.messages import SystemMessage


##################
DEFAULT_TOPN = 5
MIN_TOPN = 3
MAX_TOPN = 7
##################


class State(TypedDict):
    get_response_kwargs: Dict
    predicted_topn: int
    result: List

class Predict(BaseModel):
    top_N : int = Field("Top_N parameter value in (3,7)")


def response(state: State) -> State:
    result = get_response(**state["get_response_kwargs"])
    return {"result": result}

def predict(state: State) -> State:
    query = state["get_response_kwargs"]["query"]

    sys_prompt = (
        "You are an expert in predicting the parameters of a documents retriver given the user query.",
        "In this case, YOUR task is to predict the top_N parameter.",
        "Given the user query, extract the TOPICS of the requests and then DECIDE the optimal value of top_N.",
        "The MINIMUM value of top_N is 3 while its MAXIMUM value is 7."
        "The value 3 is set for one topic found in the user query.",
        "The value 4 is set for one or two topics found in the user query."
        "The value 5 is set for two or three topics found in the user query."
        "The value 6 is set for three or four topics found in the user query."
        "The value 7 is set for four or more topics found in the user query.",
        "Typically values 4,5, and 6 of top_n are enough but there are cases in which smaller or greater values are needed.",
        "Be precise in assigning the top_n parameter by choosing values only from the discrete INTERVAL (3,7).",
        "\nQUERY: {query}"
    )

    sys_msg = SystemMessage(content = "\n".join(sys_prompt).format(query=query))

    llm = get_chatopenai_client()
    llm = llm.with_structured_output(Predict)

    output = llm.invoke([sys_msg])
    top_n = output.top_N

    return {"predicted_topn": top_n}

def decide(state: State) -> State:
    pred_top_N = state["predicted_topn"]

    if 3 <= pred_top_N <= 7 and pred_top_N != 5:
        search_args = {"top": 5, "knn": 15}
        state["get_response_kwargs"]["search_kwargs"] = search_args
        return "new_response"
    
    return "finished"

def compile_graph():
    graph = StateGraph(State)

    graph.add_node("response_agent", response)
    graph.add_node("predict_agent", predict)

    graph.add_edge(START, "response_agent")
    graph.add_edge(START, "predict_agent")

    graph.add_conditional_edges("predict_agent", decide, {"new_response":"response_agent", "finished":END})

    graph.add_edge("response_agent", END)

    workflow = graph.compile()

    return workflow

def draw_mermaid():
    app = compile_graph()

    png_image = app.get_graph().draw_mermaid_png()

    with open("rag_agents_graph.png", "wb") as f:
        f.write(png_image)


draw_mermaid()

### USAGE 
       
# app = compile_graph()

# from mmrag import Response

# get_response_kwargs = {"query": "hello", 
#                         "sys_template": "say hi", 
#                         "messages": [], 
#                         "search_type": "text", 
#                         "openai_kwargs": {"temperature":0.1, "max_tokens":None},
#                         "output_schema": Response
#                         }

# res = app.invoke({"get_response_kwargs": get_response_kwargs})

# print(res["result"])
