from fastapi import FastAPI
import gradio as gr
import uvicorn
app = FastAPI()
@app.get("/")
def read_main():
    return {"message": "This is your main app"}
io = gr.Interface(lambda x: "Hello, " + x + "!", "textbox", "textbox")
app = gr.mount_gradio_app(app, io, path="/gradio")
if __name__ == "__main__":
    # app.launch()
    # # mounting at the root path
    uvicorn.run(
        app="prova2:app",
        # host=os.getenv("UVICORN_HOST"),  
        # port=int(os.getenv("UVICORN_PORT"))
        # port=8888,
        reload=True,
    )
