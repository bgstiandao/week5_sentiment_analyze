from transformers import pipeline

# 1. 加载情感分析pipeline（自动下载预训练模型）
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline("sentiment-analysis", model=model_name)

# 2. 预测单条文本
result1 = classifier("I love this movie! It's amazing!")
print(result1)

# 3. 预测多条文本（批量）
results = classifier([
    "I hate this product, it's terrible.",
    "The weather today is so beautiful!",
    "I'm feeling neutral about this."
])
for r in results:
    print(r)


result2 = classifier("I always lose everything!")
print(result2)