import gradio as gr

from requests_api import send_message

from server import app
import uvicorn

from rag import get_response

with gr.Blocks() as demo:
    gr.Markdown("# Chat with an LLM via FastAPI")
    chatbot = gr.Chatbot()
    state = gr.State()

    with gr.Row():
        user_input = gr.Textbox(show_label=False, placeholder="Type your message here...")
        submit_button = gr.Button("Send")
    
    with gr.Row(visible=False) as action_buttons:
        accept_button = gr.Button("Accept")
        reject_button = gr.Button("Reject")

    def handle_message(user_message, chat_history, state):
        # if "session_id" not in state:
        #     session_id = create_session()
        #     state["session_id"] = session_id
        # session_id = state.get("session_id")
        response_data = send_message(user_message)
        bot_reply = response_data.get("answer", "No response")
        chat_history.append((user_message, bot_reply))

        if response_data.get("state") == "WAITING_FOR_CONFIRMATION":
            return chat_history, state, gr.update(visible=True)
        
        return chat_history, state, gr.update(visible=False)
    
    submit_button.click(
        handle_message,
        inputs=[user_input, chatbot, state],
        outputs=[chatbot, state, action_buttons],
    )

    # def handle_accept(chat_history, state):
    #     session_id = state.get("session_id")
    #     response = accept_action(session_id)
    #     chat_history.append(("Action accepted", response.get("response")))
    #     return chat_history, state, gr.update(visible=False)
    
    # def handle_reject(chat_history, state):
    #     session_id = state.get("session_id")
    #     response = reject_action(session_id, "User rejected the action")
    #     chat_history.append(("Action rejected", response.get("response")))
    #     return chat_history, state, gr.update(visible=False)
    
    # accept_button.click(handle_accept, inputs=[chatbot, state], outputs=[chatbot, state, action_buttons])
    
    # reject_button.click(handle_reject, inputs=[chatbot, state], outputs=[chatbot, state, action_buttons])

demo.launch()


app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    # mounting at the root path
    uvicorn.run(
        app="main:app",
        # host=os.getenv("UVICORN_HOST"),  
        # port=int(os.getenv("UVICORN_PORT"))
    )