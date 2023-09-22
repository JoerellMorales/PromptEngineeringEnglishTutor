from fastapi import FastAPI,HTTPException,status

import uvicorn
from pydantic import BaseModel
from chatbot import GuidancePrompt
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Update with your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
class UserInput(BaseModel):
    role: str
    content:str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/userInput/",status_code=status.HTTP_202_ACCEPTED)
async def userInput(userPrompt:UserInput):
    model_reply = GuidancePrompt(role= userPrompt.role, content= userPrompt.content)
    return model_reply.return_model_response()

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)