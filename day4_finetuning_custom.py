import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os

# ========== 1. 创建自定义数据集（模拟你的数据） ==========
# 方式 A：从 CSV 读取（注释掉，留作参考）
# df = pd.read_csv("my_reviews.csv")
# dataset = Dataset.from_pandas(df)

# 方式 B：直接创建（演示用）
print("📦 创建自定义数据集...")

# # 1. 从 IMDb 取 80 条
# imdb_data = load_dataset("stanfordnlp/imdb", split="train")
# imdb_sample = imdb_data.shuffle(seed=42).select(range(100))
#
# # 2. 你的 20 条自定义数据
# custom_texts = [
#         "This product is amazing, I love it!",
#         "Terrible quality, broke after one day.",
#         "Not bad, works as expected.",
#         "Absolutely fantastic, highly recommend!",
#         "Waste of money, very disappointed.",
#         "Good value for the price.",
#         "Horrible customer service, never again.",
#         "Works perfectly, exactly what I needed.",
#         "Average product, nothing special.",
#         "Best purchase ever, very satisfied.",
#             #增加了样本
#         "I really liked this, it's great.",
#         "This is the worst thing I've ever bought.",
#         "It does the job, nothing more.",
#         "Fantastic! I would buy again.",
#         "Bad quality, I returned it.",
#         "Good quality and fast shipping.",
#         "Terrible, don't waste your money.",
#         "Excellent, exceeded my expectations.",
#         "Mediocre, but okay for the price.",
#         "I'm very happy with this purchase."
#     ]
#
# custom_labels = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1]  # 1=正面, 0=负面

# # 3. 合并
# all_texts = list(imdb_sample["text"]) + custom_texts
# all_labels = list(imdb_sample["label"]) + custom_labels

data = {
    "text": [
        "This product is amazing, I love it!",
        "Terrible quality, broke after one day.",
        "Not bad, works as expected.",
        "Absolutely fantastic, highly recommend!",
        "Waste of money, very disappointed.",
        "Good value for the price.",
        "Horrible customer service, never again.",
        "Works perfectly, exactly what I needed.",
        "Average product, nothing special.",
        "Best purchase ever, very satisfied.",
            #增加了样本
        "I really liked this, it's great.",
        "This is the worst thing I've ever bought.",
        "It does the job, nothing more.",
        "Fantastic! I would buy again.",
        "Bad quality, I returned it.",
        "Good quality and fast shipping.",
        "Terrible, don't waste your money.",
        "Excellent, exceeded my expectations.",
        "Mediocre, but okay for the price.",
        "I'm very happy with this purchase."
    ],
    "label": [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1]  # 1=正面, 0=负面
}


# 转换为 HuggingFace Dataset 格式
dataset = Dataset.from_dict(data)

# dataset = Dataset.from_dict({"text": all_texts, "label": all_labels})
print(f"数据集大小: {len(dataset)} 条")

# 切分训练集/测试集（80% 训练，20% 测试）
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset["train"]
test_dataset = dataset["test"]
print(f"训练集: {len(train_dataset)} 条")
print(f"测试集: {len(test_dataset)} 条")


# ========== 2. 加载分词器 ==========
# 确保路径指向你第三天保存的模型 89% 准确率的基座模型
model_path = "./my_sentiment_model"  # 或者 "./sentiment_results/checkpoint-xxx"
print(f"📥 加载基座模型: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


# ========== 3. 分词 ==========
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,    #超过512截断
        padding=False,
        max_length=128  # 自定义数据较短，用 128 足够了
    )


tokenized_train = train_dataset.map(tokenize_function, batched=True)    #batched=True的意思是一次性把数据喂给模型
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# 移除原始文本列
tokenized_train = tokenized_train.remove_columns(["text"])
tokenized_test = tokenized_test.remove_columns(["text"])

# # ========== 4. 加载模型 ==========
# model = AutoModelForSequenceClassification.from_pretrained(
#     model_name,
#     num_labels=2    #二分类
# )

# ========== 4. 评估函数 ==========
def compute_metrics(eval_pred):
    predictions, labels = eval_pred     #predictions：模型输出的原始分数，形状是 (样本数量, 分类数量)，比如 10 条样本、2 分类，就是 (10, 2)。labels：测试集里的真实标签（0 或 1）。形状是(样本数量,)
    predictions = np.argmax(predictions, axis=1)    #axis=1 表示：对每一行（每一条样本），找到这一行里最大值的列索引。
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }


# ========== 5. 🔥 核心：高级训练配置 ==========
training_args = TrainingArguments(
    output_dir="./custom_finetuned_from_89",
    num_train_epochs=3,            #增加轮数
    per_device_train_batch_size=4,  # 小数据集用更小的批次
    per_device_eval_batch_size=8,

    # 🔥 学习率调度：使用余弦退火（比线性衰减更平滑）
    learning_rate=5e-6,     # 🔥 极低学习率，只轻微调整
    lr_scheduler_type="cosine",
    warmup_steps=50,

    # 🔥 权重衰减（防止过拟合）降低权重衰减（数据少，不宜太大）
    weight_decay=0.01,

    logging_steps=2,
    eval_steps=len(train_dataset)//10,       # 🔥 每2步评估一次（每个epoch）
    save_steps=len(train_dataset)//10,
    eval_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to=[],

    # 🔥 早停：如果验证集准确率连续 2 轮不提升就停止
    save_total_limit=2,
)

# ========== 6. DataCollator ==========
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ========== 7. Trainer（加入早停回调） ==========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # 早停,耐心值为3 # 连续3轮不提升才停
)

# ========== 8. 训练 ==========
print("\n🚀 开始微调（基座模型准确率 89%）...")
trainer.train()

# ========== 9. 评估 ==========
print("\n📊 评估模型...")
eval_results = trainer.evaluate()
print(f"\n最终结果:")
print(f"准确率: {eval_results['eval_accuracy']:.4f}")
print(f"F1分数: {eval_results['eval_f1']:.4f}")

# ========== 10. 保存模型 ==========
model.save_pretrained("./custom_finetuned_model")
tokenizer.save_pretrained("./custom_finetuned_model")
print("\n✅ 微调后的模型已保存到 ./custom_finetuned_model")


# ========== 11. 预测新文本 ==========
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="./custom_finetuned_model", tokenizer="./custom_finetuned_model")

real_new_texts = [
    "It works, but it's a bit noisy.",          # 中性偏负（有点吵）
    "Good product for the price.",              # 正面（性价比高）
    "Not as good as I expected.",               # 负面（不如预期）
    "I will probably buy this again."           # 正面（会回购）
]

for text in real_new_texts:
    result = classifier(text)
    print(f"{text} -> {result[0]['label']} ({result[0]['score']:.2f})")


