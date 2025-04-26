from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
from konlpy.tag import Okt
import torch

app = FastAPI(title="KoBART Topic-Based Summary API")

# Load KoBART for summarization
tokenizer = PreTrainedTokenizerFast.from_pretrained("EbanLee/kobart-summary-v3")
model = BartForConditionalGeneration.from_pretrained("EbanLee/kobart-summary-v3")

# Load KoBERT for topic filtering (sentence embeddings)
sentence_model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
okt = Okt()

class TextInput(BaseModel):
    text: str
    topic: str  # New field for topic

@app.get("/")
async def root():
    return {"message": "KoBART Topic-Based Summary API is running"}

@app.post("/summarize")
async def summarize_text(input: TextInput):
    try:
        if not input.text.strip():
            raise HTTPException(status_code=400, detail="Text input cannot be empty")
        if not input.topic.strip():
            raise HTTPException(status_code=400, detail="Topic cannot be empty")

        # Step 1: Split text into sentences
        sentences = input.text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]

        # Step 2: Extract keywords from topic
        topic_keywords = okt.nouns(input.topic)

        # Step 3: Filter sentences related to the topic using KoBERT embeddings
        topic_embedding = sentence_model.encode(input.topic, convert_to_tensor=True)
        sentence_embeddings = sentence_model.encode(sentences, convert_to_tensor=True)
        similarities = util.cos_sim(topic_embedding, sentence_embeddings)[0]
        
        # Select sentences with high similarity (threshold: 0.5)
        relevant_sentences = [
            sentences[i] for i in range(len(sentences))
            if similarities[i] > 0.5 or any(keyword in sentences[i] for keyword in topic_keywords)
        ]

        if not relevant_sentences:
            raise HTTPException(status_code=404, detail="No sentences found related to the topic")

        # Step 4: Combine relevant sentences and summarize with KoBART
        relevant_text = ". ".join(relevant_sentences)
        inputs = tokenizer(
            relevant_text,
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

        return {"summary": summary, "relevant_sentences": relevant_sentences}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")