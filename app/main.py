from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import json
import os

load_dotenv()

app = FastAPI()

class MemoRequest(BaseModel):
    memo: str

@app.post("/memo-summary")
async def summarize_memo(req: MemoRequest):
    prompt_template = PromptTemplate.from_template("""
You are an assistant that summarizes caregiving memos.

Given the following caregiving memo, extract and summarize the information into a JSON object
with the following categories: "health", "medic", "meal", "walk", "comm", and "toilet".

Each value in the JSON should be 1–2 concise sentences in Korean.

If a category is not mentioned in the memo, return the string "정보 없음" for that category.

Respond ONLY with a valid JSON. Do not explain anything.

Memo:
{memo}
""")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )

    chain = prompt_template | llm

    result = chain.invoke({"memo": req.memo})
    parsed = json.loads(result.content)
    return {"summary": parsed}

