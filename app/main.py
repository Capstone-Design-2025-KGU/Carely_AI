from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, BartForConditionalGeneration
import re
from typing import Dict

app = FastAPI()

# KoBART 모델 로드
model_name = "EbanLee/kobart-summary-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)


# 입력 데이터 모델
class PatientData(BaseModel):
    notes: str


# 텍스트 분류를 위한 개선된 함수
def classify_notes(notes: str) -> Dict[str, str]:
    categories = {
        "health": "",
        "medic": "",
        "meal": "",
        "walk": "",
        "comm": "",
        "toilet": ""
    }

    # 문장 분리 및 정규화
    sentences = [s.strip() for s in re.split(r'[.!?]', notes) if s.strip()]

    # 키워드 기반 분류 (더 구체적인 키워드와 우선순위 설정)
    for sentence in sentences:
        if any(keyword in sentence for keyword in ["복용", "약", "투약", "약물"]):
            categories["medic"] = sentence
        elif any(keyword in sentence for keyword in ["식사", "먹", "식욕", "밥"]):
            categories["meal"] = sentence
        elif any(keyword in sentence for keyword in ["산책", "운동", "활동", "걷"]):
            categories["walk"] = sentence
        elif any(keyword in sentence for keyword in ["기분", "우울", "정서", "감정"]):
            categories["comm"] = sentence
        elif any(keyword in sentence for keyword in ["배변", "배뇨", "화장실", "똥", "오줌", "대변", "소변"]):
            categories["toilet"] = sentence
        else:
            categories["health"] = sentence  # 명시적 키워드 없으면 건강으로

    return categories


# 요약 생성 (중복 방지 및 품질 개선)
def summarize_text(text: str, category: str) -> str:
    if not text:
        if category == "health":
            return "환자의 전반적인 건강 상태는 안정적이다."
        return "정보 없음."

    # 입력 텍스트 정규화 (중복 단어 제거)
    text = " ".join(dict.fromkeys(text.split()))  # 단어 단위 중복 제거

    # KoBART 요약
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=50,
        min_length=10,
        num_beams=4,
        length_penalty=1.0,
        repetition_penalty=2.0,  # 중복 단어 억제
        no_repeat_ngram_size=3,  # 3-gram 반복 방지
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


@app.post("/summarize")
async def summarize_patient_data(data: PatientData):
    # 1. 텍스트 분류
    categorized = classify_notes(data.notes)

    # 2. 각 분야별 요약
    summaries = {}
    for category, text in categorized.items():
        summaries[category] = summarize_text(text, category)

    # Return summaries directly without the "summaries" key
    return summaries


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)