from fastapi import FastAPI
import gradio as gr
import uvicorn

from mmrag import SYSTEM_MESSAGE_TEMPLATE, Response
from utils.azure_utils import get_response


### FASTAPI BACKEND ###

app = FastAPI()

@app.get("/status")
async def health_check():
    return {"status":"UP"}


### GRADIO FRONTEND ###

def gradio_bot_response(query, chat, state):
    if not state.get("history", None): ### REAL CHAT HISTORY
        state["history"] = []

    response, sources, _ = get_response(query=query, 
                                        sys_template=SYSTEM_MESSAGE_TEMPLATE, 
                                        messages=state["history"], # remove citations from chat history
                                        search_type="vector_text", 
                                        openai_kwargs={"temperature":0.1, "max_tokens":None},
                                        output_schema=Response) # LLM response + retrieved sources

    # extract results from output schema
    related, intent, answer = response.related, response.intent, response.answer

    chat.append(gr.ChatMessage(role="user", content=query))
    chat.append(gr.ChatMessage(role="assistant", content=answer))

    if related:
        chat.append(
                    gr.ChatMessage(
                        role="assistant",
                        content="The user query is related to the knowledge base.",
                        metadata={"title": "✅ Related", "status":"done"}))
        chat.append(
                    gr.ChatMessage(
                        role="assistant",
                        content="\n".join([f"• {x}" for x in intent]),
                        metadata={"title": "🎯 Intent", "status":"done"}))
        chat.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=sources,
                        metadata={"title": "📚 Citations", "status":"done"}))
        
        ### REAL CHAT HISTORY UPDATE
        user_msg = {"role": "user", "content": f"{query}"}
        bot_msg = {"role": "assistant", "content": f"{answer}"}

        state["history"].extend([user_msg, bot_msg])

    else:
        chat.append(
                    gr.ChatMessage(
                        role="assistant",
                        content="The user query is NOT related to the knowledge base.",
                        metadata={"title": "❌ Related", "status":"done"}))

    return "", chat, state # first output clears the user input Textbox


with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    gr.Markdown("# RAG with FastAPI + Gradio")
    chatbot = gr.Chatbot(type="messages")
    state = gr.State(value={})

    user_input = gr.Textbox(show_label=False, placeholder="Type your message here...", submit_btn=True)
    
    user_input.submit(
        gradio_bot_response,
        inputs=[user_input, chatbot, state],
        outputs=[user_input, chatbot, state],
        show_progress_on=[chatbot, state],
    )
    

app = gr.mount_gradio_app(app, demo, path="/")



if __name__ == "__main__":
    uvicorn.run(
        app="app:app",
        reload=True,
    )
