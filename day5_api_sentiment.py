from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from typing import List
import uvicorn

# ========== 1. 加载微调好的模型 ==========
print("🤖 加载情感分析模型...")
# classifier = pipeline(
#     "sentiment-analysis",
#     model="./custom_finetuned_model",
#     tokenizer="./custom_finetuned_model"
# )

#加载hugging face上的上传的模型
classifier = pipeline(
    "sentiment-analysis",
    model="bgs-123/custom_finetuned_model",
    tokenizer="bgs-123/custom_finetuned_model",
    model_kwargs={"low_cpu_mem_usage": True}
)

print("✅ 模型加载成功！")


# ========== 2. 定义请求/响应数据结构 ==========
class TextRequest(BaseModel):
    text: str


class BatchTextRequest(BaseModel):
    texts: List[str]


class PredictionResponse(BaseModel):
    label: str
    confidence: float


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]


# ========== 3. 创建 FastAPI 应用 ==========
app = FastAPI(
    title="情感分析 API",
    description="基于 DistilBERT 微调的情感分析服务（正面/负面）",
    version="1.0"
)


@app.get("/")
def read_root():
    return {
        "message": "欢迎使用情感分析 API！",
        "endpoints": {
            "/predict": "单条文本预测（POST）",
            "/predict_batch": "批量文本预测（POST）",
            "/docs": "交互式文档"
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    """
    预测单条文本的情感（正面/负面）
    """
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="文本不能为空")

    # 预测
    result = classifier(request.text)[0]
    return {
        "label": result["label"],
        "confidence": round(result["score"], 4)
    }


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchTextRequest):
    """
    批量预测多条文本的情感
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="文本列表不能为空")

    # 批量预测
    results = classifier(request.texts)
    return {
        "results": [
            {
                "label": r["label"],
                "confidence": round(r["score"], 4)
            }
            for r in results
        ]
    }


# ========== 4. 健康检查 ==========
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# ========== 5. 启动服务 ==========
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)