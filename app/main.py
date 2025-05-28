from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, BartForConditionalGeneration
import re
from typing import Dict, List
import torch

app = FastAPI()

# 모델 로딩 (1회만)
MODEL_NAME = "EbanLee/kobart-summary-v3"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

# 요청 데이터 구조
class PatientData(BaseModel):
    notes: str

# 카테고리 키워드 정의
CATEGORY_KEYWORDS = {
    "medic": ["복용", "약", "투약", "약물"],
    "meal": ["식사", "먹", "식욕", "밥"],
    "walk": ["산책", "운동", "활동", "걷"],
    "comm": ["기분", "우울", "정서", "감정"],
    "toilet": ["배변", "배뇨", "화장실", "똥", "오줌", "대변", "소변"]
}

DEFAULT_CATEGORIES = ["health", "medic", "meal", "walk", "comm", "toilet"]

# 문장 분류
def classify_notes(notes: str) -> Dict[str, List[str]]:
    categorized = {cat: [] for cat in DEFAULT_CATEGORIES}
    sentences = [s.strip() for s in re.split(r'[.!?\n]', notes) if s.strip()]
    seen = set()

    for sentence in sentences:
        if sentence in seen:
            continue
        seen.add(sentence)

        matched = False
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(k in sentence for k in keywords):
                categorized[category].append(sentence)
                matched = True

        if not matched:
            categorized["health"].append(sentence)

    return categorized

# 텍스트 요약
def summarize_text(text_list: List[str], category: str) -> str:
    if not text_list:
        return "환자의 전반적인 건강 상태는 안정적입니다." if category == "health" else "정보 없음."

    text = " ".join(text_list)
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=50,
            min_length=10,
            num_beams=4,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        return f"{category} 요약 중 오류 발생: {str(e)}"

@app.post("/summarize")
async def summarize_patient_data(data: PatientData):
    categorized = classify_notes(data.notes)
    summaries = {cat: summarize_text(texts, cat) for cat, texts in categorized.items()}
    return summaries