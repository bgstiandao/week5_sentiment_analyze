import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification, # 负责加载预训练模型 + 自动加上分类头
    Trainer,    # 负责训练循环（帮你自动做前向、反向、更新参数）
    TrainingArguments,  # 负责训练参数的配置（学几轮、多大批次等）
    DataCollatorWithPadding # 负责把不同长度的句子填充到一样长
)
from datasets import load_dataset       ## 负责帮你下载数据集
import numpy as np
from sklearn.metrics import accuracy_score, f1_score    #accuracy_score（准确率）和 f1_score（F1分数）

# ========== 1. 加载数据集 ==========
# 使用 IMDb 电影评论数据集（5万条标注好的影评）
print("📦 加载 imdb 数据集...")
dataset = load_dataset("stanfordnlp/imdb")

# 只取 2000 条训练，500 条测试（为了快速演示）
train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))   #.select(range(2000))：只取前 2000 条作为训练集（为了让你训练快一点），取 500 条作为测试集。
test_dataset = dataset["test"].shuffle(seed=42).select(range(500))

print(f"训练集: {len(train_dataset)} 条")
print(f"测试集: {len(test_dataset)} 条")

# 查看数据样例
print(f"\n样例: {train_dataset[0]['text'][:100]}...")
print(f"标签: {train_dataset[0]['label']} (0=负面, 1=正面)")

# ========== 2. 加载分词器 ==========
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ========== 3. 数据预处理（分词） ==========
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,    # 如果超过 512 个字，自动截断
        padding=False,      # 先不填充，后面用 DataCollator 统一处理
        max_length=512
    )

print("\n🔧 正在分词...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)    #.map()：对数据集里的每一条评论，都执行一次 tokenize_function，把文字全变成数字ID。
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# 删掉原始的英文字段，只保留数字，节省内存。
tokenized_train = tokenized_train.remove_columns(["text"])
tokenized_test = tokenized_test.remove_columns(["text"])

# ========== 4. 加载预训练模型（带分类头） ==========
print("\n🤖 加载模型...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2  # 加上一个分类头，输出 2 个值（正面 或 负面）
)


# ========== 5. 定义评估指标 ==========
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)    # 模型输出两个数字，哪个大就认为模型猜的是哪个类别。（0或1）
    acc = accuracy_score(labels, predictions)   # 计算猜对的比例
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}


# ========== 6. 配置训练参数 ==========
training_args = TrainingArguments(
    output_dir="./sentiment_results",      # 保存目录
    num_train_epochs=3,                    # 训练轮数
    per_device_train_batch_size=16,        # 训练批次大小，一次喂给模型 16 条数据
    per_device_eval_batch_size=64,         # 评估批次大小，测试时一次喂 64 条
    warmup_steps=100,                      # 预热步数，前100步慢慢提高学习率（热身）
    weight_decay=0.01,                     # 权重衰减，防止过拟合（别让模型死记硬背）
    logging_steps=50,                      # 日志间隔，每50步打印一次进度
    eval_steps=200,                        # 评估间隔，每200步在测试集上测一次
    save_steps=200,                        # 保存间隔，每200步保存一次模型
    eval_strategy="steps",                  # 按步数评估
    load_best_model_at_end=True,           # 训练结束加载最佳模型
    metric_for_best_model="accuracy",      # 用准确率判断哪个模型最好
    report_to=[]                            # 不汇报到外部服务
)

# ========== 7. 创建 DataCollator ==========
# 自动将不同长度的序列填充到相同长度，GPU 要求一次性输入的数据必须形状一样。
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# ========== 8. 创建 Trainer 并训练 ==========
trainer = Trainer(      #Trainer 是一个超级傻瓜式工具。你把模型、数据、设置全扔给它，它自动帮你跑循环：
    model=model,                        # 把大脑放进去
    args=training_args,                 # 把菜谱放进去
    train_dataset=tokenized_train,      # 把训练数据放进去
    eval_dataset=tokenized_test,        # 把测试数据放进去
    data_collator=data_collator,        # 把填充器放进去
    compute_metrics=compute_metrics,    # 把评分员放进去
)
print("\n🚀 开始训练...")
trainer.train()

# ========== 9. 评估模型 ==========
print("\n📊 评估模型...")
eval_results = trainer.evaluate()
print(f"\n最终评估结果:")
print(f"准确率: {eval_results['eval_accuracy']:.4f}")
print(f"F1分数: {eval_results['eval_f1']:.4f}")

# ========== 10. 保存模型 ==========
model.save_pretrained("./my_sentiment_model")
tokenizer.save_pretrained("./my_sentiment_model")
print("\n✅ 模型已保存到 ./my_sentiment_model")

# ========== 11. 用自己训练的模型做预测 ==========
print("\n🔮 使用训练好的模型进行预测...")

# 加载保存的模型
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="./my_sentiment_model", tokenizer="./my_sentiment_model")

test_texts = [
    "This movie was absolutely fantastic! I loved every minute.",
    "Terrible film, waste of time and money.",
    "It was okay, nothing special but not bad."
]

for text in test_texts:
    result = classifier(text)
    print(f"\n文本: {text}")
    print(f"预测: {result[0]['label']} (置信度: {result[0]['score']:.4f})")
