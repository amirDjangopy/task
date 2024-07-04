#این کد به شما کمک می‌کند تا شباهت معنایی بین دو متن را با استفاده از مدل BERT محاسبه کنید. به طور کلی، کد شامل مراحل زیر است:

1. نصب کتابخانه‌ها
ابتدا دو کتابخانه transformers و torch نصب می‌شوند که برای استفاده از مدل‌های پردازش زبان طبیعی (NLP) و انجام محاسبات عددی به کار می‌روند.

python
Copy code
!pip install transformers torch
2. بارگیری مدل و توکنایزر
از کتابخانه transformers استفاده می‌شود تا مدل و توکنایزر BERT بارگیری شوند. برای این کار نیاز به یک توکن احراز هویت از سایت HuggingFace دارید که در متغیر token قرار داده می‌شود.

python
Copy code
from transformers import BertModel, BertTokenizer
import torch
import numpy as np

token = "YOUR_HUGGINGFACE_TOKEN"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_auth_token=token)
model = BertModel.from_pretrained("bert-base-uncased", use_auth_token=token)

print(model)
print(tokenizer)
3. محاسبه شباهت کسینوسی
تابع cosine_similarity شباهت کسینوسی بین دو بردار را محاسبه می‌کند. شباهت کسینوسی یک معیار اندازه‌گیری است که برای تعیین میزان شباهت دو بردار استفاده می‌شود.

python
Copy code
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
4. استخراج بردارهای BERT
تابع get_bert_embeddings متن ورودی را به بردارهای BERT تبدیل می‌کند. این تابع از توکنایزر برای تبدیل متن به توکن‌ها و از مدل برای استخراج بردارهای BERT استفاده می‌کند.

python
Copy code
def get_bert_embeddings(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings
5. محاسبه BERTScore
تابع calculate_bertscore بردارهای BERT دو متن را محاسبه کرده و سپس شباهت کسینوسی بین آن‌ها را به عنوان BERTScore بازمی‌گرداند.

python
Copy code
def calculate_bertscore(context, response, model, tokenizer):
    context_vector = get_bert_embeddings(context, model, tokenizer)
    response_vector = get_bert_embeddings(response, model, tokenizer)
    score = cosine_similarity(context_vector, response_vector)
    return score
6. مثال از داده‌ها
یک مثال از متون برای محاسبه BERTScore آورده شده و نتیجه آن چاپ می‌شود.

python
Copy code
context = "This is an example context."
response = "This is an example response."

score = calculate_bertscore(context, response, model, tokenizer)
print("BERTScore:", score)
نتیجه نهایی
کد نهایی یک نمره (BERTScore) برمی‌گرداند که نشان‌دهنده میزان شباهت معنایی بین دو متن ورودی است. این نمره از محاسبه شباهت کسینوسی بین بردارهای معنایی استخراج شده توسط مدل BERT به دست می‌آید.

این کد به طور کامل توضیح داده شد که چه کاری انجام می‌دهد و چگونه شباهت معنایی بین دو متن را محاسبه می‌کند. اگر سوال دیگری دارید یا نیاز به توضیحات بیشتر دارید، بپرسید.










ChatGPT can make mistakes. Check important i
