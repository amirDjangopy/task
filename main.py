from transformers import BertModel, BertTokenizer
import torch
import numpy as np

# تنظیم توکن HuggingFace
token = "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"

# بارگیری مدل و توکنایزر BERT با استفاده از توکن
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_auth_token=token)
model = BertModel.from_pretrained("bert-base-uncased", use_auth_token=token)

# نمایش مدل و توکنایزر برای اطمینان از بارگیری صحیح
print(model)
print(tokenizer)

# تابع محاسبه شباهت کسینوسی
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# تابع استخراج بردارهای BERT برای متن ورودی
def get_bert_embeddings(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

# تابع محاسبه BERTScore
def calculate_bertscore(context, response, model, tokenizer):
    context_vector = get_bert_embeddings(context, model, tokenizer)
    response_vector = get_bert_embeddings(response, model, tokenizer)
    score = cosine_similarity(context_vector, response_vector)
    return score

# مثال از داده‌ها
context = "This is an example context."
response = "This is an example response."

# محاسبه BERTScore
score = calculate_bertscore(context, response, model, tokenizer)
print("BERTScore:", score)