import os
import json
import requests
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import xmltodict
import google.generativeai as palm

app = FastAPI(title="IFPyAPI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Google(BaseModel):
    query: str
    language: str
    country: str
    ceid: str

@app.post("/hotnews")
async def get_google_news(google: Google):
    FEED = os.environ['FEED']
    url = f'{FEED}/search?q={google.query}&hl={google.language}&gl={google.country}&ceid={google.ceid}'
    res = requests.get(url)
    if res.status_code != 200:
        raise HTTPException(status_code=500)
    rss_data = xmltodict.parse(res.text)
    channel = rss_data['rss']['channel']
    items = channel['item']
    return items

class Answer(BaseModel):
    content: str
    question: str

@app.post("/answer")
async def get_answer(answer: Answer):
    PALM = os.environ['PALM']
    palm.configure(api_key=PALM)
    models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
    model = models[0].name
    prompt = f"""Context:{answer.content}
    Q:{answer.question}
    A:
    """
    completion = palm.generate_text(
        model=model,
        prompt=prompt,
        temperature=0,
        candidate_count=1,
        top_k=1,
        top_p=0.95,
        max_output_tokens=1024
    )
    return completion.result
    
