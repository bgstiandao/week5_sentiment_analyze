from transformers import AutoTokenizer

# ========== 1. 加载分词器 ==========
# 使用与昨天情感分析相同的模型的分词器
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"✅ 分词器加载完成，词汇表大小: {tokenizer.vocab_size}")

# ========== 2. 最简单的分词 ==========
text = "I love this movie!"
tokens = tokenizer.tokenize(text)
print(f"\n原始文本: {text}")
print(f"分词结果: {tokens}")

# ========== 3. 文本 → 数字ID ==========
input_ids = tokenizer.encode(text)
print(f"\n文本转ID: {input_ids}")

# ========== 4. 数字ID → 还原文本 ==========
decoded = tokenizer.decode(input_ids)
print(f"ID还原文本: {decoded}")

# ========== 5. 批量处理（加填充和截断） ==========
texts = [
    "I love this movie!",
    "This is the worst film I have ever seen.",
    "It was okay, not great."
]

# 编码：自动填充到相同长度，并返回 PyTorch 张量
encoded = tokenizer(
    texts,
    padding=True,           # 填充到相同长度
    truncation=True,        # 超长截断（模型有最大长度限制）
    #max_length=128,      # 最大长度（可选）
    return_tensors="pt"     # 返回 PyTorch 张量
)

print(f"\n批量编码结果:")
print(f"input_ids 形状: {encoded['input_ids'].shape}")
print(f"attention_mask 形状: {encoded['attention_mask'].shape}")
print(f"\ninput_ids 内容:\n{encoded['input_ids']}")       #[CLS] (101) = 分类标记，放在开头,[SEP] (102) = 分隔标记，放在结尾
print(f"\nattention_mask 内容:\n{encoded['attention_mask']}")     #attention_mask 中 1 代表真实文本，0 代表填充位置

# ========== 6. 解码查看填充效果 ==========
for i, ids in enumerate(encoded['input_ids']):
    decoded_text = tokenizer.decode(ids)
    print(f"\n样本 {i+1}: {decoded_text}")



# 1. 对比不同句子的 token 数量
texts = ["short", "a very much longer sentence with many words"]
for t in texts:
    print(f"{t}: {len(tokenizer.encode(t))} tokens")

# 2. 查看特殊标记位置
ids = tokenizer.encode("hello world")
print(f"第一个token: {ids[0]} -> {tokenizer.decode([ids[0]])}")    #第一个token: 101 -> [CLS]
print(f"最后一个token: {ids[-1]} -> {tokenizer.decode([ids[-1]])}") #最后一个token: 102 -> [SEP]