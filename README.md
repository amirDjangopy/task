## قدم به قدم اجرای کد با توضیحات کامل

1 نصب کتابخانه‌ها
ابتدا باید کتابخانه‌های transformers و torch را نصب کنید. این کتابخانه‌ها برای کار با مدل‌های پردازش زبان طبیعی (NLP) و انجام محاسبات عددی استفاده می‌شوند.

```bash
pip install transformers 
```

3. وارد کردن کتابخانه‌ها
ابتدا کتابخانه‌های مورد نیاز را وارد می‌کنیم.
```bash
from transformers import BertModel, BertTokenizer
import torch
import numpy as np
BertModel و BertTokenizer برای بارگیری مدل BERT و توکنایزر آن استفاده می‌شوند.
torch برای انجام محاسبات عددی استفاده می‌شود.
numpy برای انجام عملیات ریاضیاتی مانند محاسبه شباهت کسینوسی استفاده می‌شود.
```
<br></br>
4. بارگیری مدل و توکنایزر
یک توکن دسترسی از HuggingFace نیاز دارید که باید آن را جایگزین YOUR_HUGGINGFACE_TOKEN کنید. سپس مدل و توکنایزر BERT را بارگیری می‌کنیم.


```bash 
token = "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" توکنی که از سایت HuggingFace گرفتم
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_auth_token=token)
model = BertModel.from_pretrained("bert-base-uncased", use_auth_token=token)
BertTokenizer.from_pretrained توکنایزر BERT را بارگیری می‌کند.
BertModel.from_pretrained مدل BERT را بارگیری می‌کند.
```
  توکن 

  ```bash
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
```

5. تعریف تابع محاسبه شباهت کسینوسی
این تابع شباهت کسینوسی بین دو بردار را محاسبه می‌کند.

```bash 
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
np.dot(a, b) ضرب نقطه‌ای دو بردار را محاسبه می‌کند.
np.linalg.norm(a) و np.linalg.norm(b) نُرم (طول) هر بردار را محاسبه می‌کنند.
```


6. تعریف تابع استخراج بردارهای BERT
این تابع متن ورودی را به بردارهای BERT تبدیل می‌کند.


```bash 
def get_bert_embeddings(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings
tokenizer(text, return_tensors="pt", padding=True, truncation=True) متن را توکنیزه کرده و به فرمت قابل استفاده برای مدل تبدیل می‌کند.
torch.no_grad() محاسبات را بدون محاسبه گرادیان‌ها انجام می‌دهد (برای صرفه‌جویی در حافظه).
outputs.last_hidden_state.mean(dim=1).squeeze().numpy() خروجی آخرین لایه مدل را استخراج کرده و به numpy array تبدیل می‌کند.
```

7. تعریف تابع محاسبه BERTScore
این تابع بردارهای BERT دو متن را محاسبه کرده و سپس شباهت کسینوسی بین آن‌ها را به عنوان BERTScore بازمی‌گرداند.


```bash
def calculate_bertscore(context, response, model, tokenizer):
    context_vector = get_bert_embeddings(context, model, tokenizer)
    response_vector = get_bert_embeddings(response, model, tokenizer)
    score = cosine_similarity(context_vector, response_vector)
    return score
get_bert_embeddings(context, model, tokenizer) بردارهای BERT متن زمینه را استخراج می‌کند.
get_bert_embeddings(response, model, tokenizer) بردارهای BERT پاسخ را استخراج می‌کند.
cosine_similarity(context_vector, response_vector) شباهت کسینوسی بین دو بردار را محاسبه می‌کند.
```


8. تعریف تابع محاسبه شباهت جاکارد
این تابع شباهت جاکارد بین دو متن را محاسبه می‌کند.
```bash
def jaccard_similarity(query, document):
    query_set = set(query.lower().split())
    document_set = set(document.lower().split())
    intersection = query_set.intersection(document_set)
    union = query_set.union(document_set)
    return len(intersection) / len(union)
query.lower().split() متن پرسش را به کلمات جداگانه تقسیم و به حروف کوچک تبدیل می‌کند.
set(query.lower().split()) مجموعه‌ای از کلمات منحصر به فرد را ایجاد می‌کند.
intersection اشتراک دو مجموعه را محاسبه می‌کند.
union اتحاد دو مجموعه را محاسبه می‌کند.
len(intersection) / len(union) نسبت تعداد کلمات مشترک به تعداد کل کلمات را محاسبه می‌کند.
```


9. اجرای برنامه
در نهایت، برنامه را با نمونه‌ای از داده‌ها اجرا می‌کنیم.

```bash
if name == "main":
    context = "This is an example context."
    response = "This is an example response."
    
    bert_score = calculate_bertscore(context, response, model, tokenizer)
    jaccard_score = jaccard_similarity(context, response)
    
    print("BERTScore:", bert_score)
    print("Jaccard Similarity:", jaccard_score)
calculate_bertscore(context, response, model, tokenizer) نمره BERTScore را محاسبه می‌کند.
jaccard_similarity(context, response) شباهت جاکارد را محاسبه می‌کند.
print("BERTScore:", bert_score) و print("Jaccard Similarity:", jaccard_score) نتایج را چاپ می‌کنند.
```

اجرای کد
ابتدا مطمئن شوید که کتابخانه‌های مورد نیاز را نصب کرده‌اید.
توکن دسترسی HuggingFace را در کد وارد کنید.




این کد نتیجه محاسبه BERTScore و شباهت جاکارد بین دو متن نمونه را چاپ می‌کند.
