from transformers import pipeline
import torch

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

pipe = pipeline("sentiment-analysis", model=model_name, tokenizer=tokenizer_name, device=-1)
text = "This is a test message."
out = pipe(text)
print(f"Pipeline output for '{text}': {out}")

model_raw = pipe.model
inputs = pipe.tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model_raw(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    
print(f"Direct Model Logits: {logits}")
print(f"Direct Model Probs (Softmax): {probs}")
