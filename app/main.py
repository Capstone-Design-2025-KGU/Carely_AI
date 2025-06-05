import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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

Respond ONLY with a valid JSON object (without any markdown code block or explanation).

Memo:
{memo}
""")

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )

    chain = prompt_template | llm

    result = chain.invoke({"memo": req.memo})
    print("LLM result:", result.content)

    # 코드블록 제거
    content = result.content.strip()
    content = re.sub(r"^```json", "", content)
    content = re.sub(r"\n?```$", "", content)
    content = content.strip()

    try:
        parsed = json.loads(content)
    except Exception as e:
        print("JSON parsing error:", e)
        print("Content:", content)
        raise HTTPException(status_code=500, detail="Invalid JSON format from LLM response")

    return parsed
