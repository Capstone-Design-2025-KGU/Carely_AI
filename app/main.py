from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

app = FastAPI(title="KoBART Summary API")

# Load KoBART model
tokenizer = PreTrainedTokenizerFast.from_pretrained("EbanLee/kobart-summary-v3")
model = BartForConditionalGeneration.from_pretrained("EbanLee/kobart-summary-v3")

class TextInput(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "KoBART Summary API is running"}

@app.post("/summarize")
async def summarize_text(input: TextInput):
    try:
        if not input.text.strip():
            raise HTTPException(status_code=400, detail="Text input cannot be empty")
        inputs = tokenizer(
            input.text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=150,
            min_length=40,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")