# 🧠 第五周：Transformer 情感分析实战

本项目基于 HuggingFace DistilBERT，实现了情感分析模型的训练、微调、API 封装及云端部署。

## 📁 项目结构

week5_nlp/
├── day1_pipeline_demo.py # Pipeline 体验
├── day2_tokenizer_demo.py # Tokenizer 深入理解
├── day3_train_sentiment.py # 手动训练（IMDb）
├── day4_finetuning_custom.py # 自定义数据微调
├── day5_sentiment_api.py # FastAPI 封装
├── custom_finetuned_model/ # 微调后的模型（已上传 HF）
├── requirements.txt # 依赖列表
└── README.md

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 启动本地 API

bash

```
python day5_sentiment_api.py
```



访问 `http://127.0.0.1:8000/docs` 查看交互式文档。

### 3. 测试预测

bash

```
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```



## 🧠 模型托管

模型已上传至 Hugging Face：

- 模型地址：`你的用户名/你的情感分析模型`
- 加载方式：

python

```
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="你的用户名/你的情感分析模型")
```



## 📊 模型性能

| 指标         | 数值          |
| :----------- | :------------ |
| 训练数据     | IMDb 100 条   |
| 微调数据     | 自定义 20 条  |
| 微调后准确率 | 100%          |
| 推理速度     | < 50ms（CPU） |

## 🔧 技术栈

- **HuggingFace Transformers**：预训练模型 & Pipeline
- **FastAPI**：Web 框架
- **PyTorch**：深度学习框架
- **scikit-learn**：评估指标

## 🌐 部署状态

- ☑ 

  本地 API 运行成功

- ☑ 

  模型上传至 Hugging Face

- □ 

  云端部署（Render 内存不足，待优化后重试）

## 🔮 可扩展方向

- □ 

  中文情感分析（`bert-base-chinese`）

- □ 

  多分类（如 5 星评分）

- □ 

  缓存机制（Redis）

- □ 

  异步批量请求

## 📝 学到的关键技能

1. Pipeline 极速调用预训练模型
2. Tokenizer 文字 ↔ 数字转换
3. Trainer 训练循环自动化
4. 迁移学习与微调技巧
5. FastAPI 封装模型服务
6. Hugging Face 模型托管与分享

## 👤 作者

[bgstiandao] 