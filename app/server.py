from pydantic import BaseModel, Field
from fastapi import FastAPI, Request, Response, status, HTTPException
import uvicorn
from rag import get_response


app = FastAPI()


class Client(BaseModel):
    question: str


history = [{
            "role": "system",
            "content": "You are an expert report analyzer that provide relevant information to users querying it",
    }]


@app.get("/")
async def health_check(request: Request):
    return {"status":"OK"}

@app.post("/chat")
async def chat(client: Client, response: Response):
    global history

    messages = history + [{"role": "user", "content": client.question}]

    try:
        response = get_response(messages)
        answer = response.choices[0].message.content
        intent = response.choices[0].message.context["intent"]
        citations = response.choices[0].message.context["citations"]
        history = messages + [{"role": "assistant", "content": answer}]
        return {"answer":answer, "intent":intent, "citations":citations}
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return HTTPException(detail=str(e), status_code=500)


if __name__ == '__main__':
    uvicorn.run('server:app', host='127.0.0.1', port=8000, reload=True)
