from fastapi import FastAPI
from pydantic import BaseModel
from app.predict import predict_model


app = FastAPI()


@app.get("/")
def root():
    return "credit classification modle"

# 텍스트 분류를 처리하는 엔드포인트
class DataClassificationRequest(BaseModel):
    data: list

# 예측
@app.post("/predict")
async def predict(request: DataClassificationRequest):
    list_data = request.data
    print(list_data)
    prediction = predict_model(list_data)
    return {prediction}