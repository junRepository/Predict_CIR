from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_model


app = FastAPI()


@app.get("/")
def root():
    return "credit classification modle"

# 텍스트 분류를 처리하는 엔드포인트
class DataClassificationRequest(BaseModel):
    data: list


@app.post("/predict")
async def predict(request: DataClassificationRequest):
    text = request.data
    prediction = predict_model(text)
    return {"prediction": prediction}