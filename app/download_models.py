from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer

# KoBART 모델 및 토크나이저 다운로드
print("Downloading KoBART tokenizer...")
tokenizer = PreTrainedTokenizerFast.from_pretrained("EbanLee/kobart-summary-v3")
print("Downloading KoBART model...")
model = BartForConditionalGeneration.from_pretrained("EbanLee/kobart-summary-v3")

# KoBERT (SentenceTransformer) 모델 다운로드
print("Downloading KoBERT model...")
sentence_model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

print("All models downloaded successfully!")