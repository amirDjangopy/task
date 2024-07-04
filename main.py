from transformers import BertModel, BertTokenizer
import torch
import numpy as np

# بارگیری مدل و توکنایزر
token = "YOUR_HUGGINGFACE_TOKEN"  # جایگزین کنید با توکن خود
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_auth_token=token)
model = BertModel.from_pretrained("bert-base-uncased", use_auth_token=token)

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

# تابع محاسبه شباهت جاکارد
def jaccard_similarity(query, document):
    query_set = set(query.lower().split())
    document_set = set(document.lower().split())
    intersection = query_set.intersection(document_set)
    union = query_set.union(document_set)
    return len(intersection) / len(union)

if __name__ == "main":
    context = "This is an example context."
    response = "This is an example response."
    
    bert_score = calculate_bertscore(context, response, model, tokenizer)
    jaccard_score = jaccard_similarity(context, response)
    
    print("BERTScore:", bert_score)
    print("Jaccard Similarity:", jaccard_score)
