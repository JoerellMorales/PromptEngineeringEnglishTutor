from fastapi import FastAPI,HTTPException,status

import uvicorn
from pydantic import BaseModel,Field
from chatbot import GuidancePrompt,GuardRail
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import guidance
import json
from typing import List,Dict

load_dotenv()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Update with your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
class Message(BaseModel):
    sender: str
    content: str

class History(BaseModel):
    message_history : List[Message]
    name : str
    proficiency : str
    language : str

class Translate(BaseModel):
    content: str
    proficiency : str
    language : str

class Vocab(BaseModel):
    sender: str
    content : str
    proficiency : str

@app.get("/")
async def root():
    return {"message": "Hello World"}

#1st Endpoint
@app.post("/history/",status_code=status.HTTP_202_ACCEPTED)
async def testInput(userPrompt:History):
    prompt = GuidancePrompt(message_history=userPrompt.message_history)
    return prompt.return_model_response()


#Second Endpoint
@app.post("/translate/",status_code=status.HTTP_202_ACCEPTED)
async def translate_response(promptTotranslate:Translate):
    prompt = GuardRail(message=promptTotranslate.content,
                         proficiency=promptTotranslate.proficiency,
                         language=promptTotranslate.language)
    
    return prompt.translate()


#Third Endpoint
@app.post("/vocab/",status_code=status.HTTP_202_ACCEPTED)
async def generate_vocab(message:Vocab):
    prompt = GuardRail(message=message.content,
                         proficiency=message.proficiency,
                         sender=message.sender)
    
    return prompt.vocab()

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)