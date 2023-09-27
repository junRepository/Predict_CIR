## FastAPI와 Docker를 이용하여 Predict하는 모델 만들기 
<img src="https://github.com/junRepository/Predict_CIR/assets/97268718/5f3bafea-0992-4b3a-995d-835a2ff917a6" width="100%"> 
- 프로젝트의 구성도는 위와 같다.
- FastAPI를 통해 Pytorh 모델을 API식으로 빌드하고 Docker를 활용하여 배포한다.
- 이때 배포하는 동안에 컴퓨터를 계속 켜있을 수 없으므로 GCP(Google Cloud Platform)을 이용하여 계속 배포할 수 있게 한다.
- web page에서 사용자가 입력하는 데이터를 전달받아 에측을 하고 예측값을 web page에 전달하여 사용자들이 쉽게 건전 / 비건전을 확인할 수 있게 하였다.<br/>

***

## 1. Pytorch 모델 FastAPI 빌드

1. ##### 학습된 모델 저장 (.pth)
    * `torch.save`를 이용하여 .pth형태로 모델 학습에 사용된 텐서(Tensor)와 매개변수를 저장한다.<br/>
    * .pth 파일을 보면 아래 그림과 같이 각 레이어들의 가중치(weight)와 편향(bias)이 텐서 형태로 저장되어있다.
    *  학습된 모델을 기반으로 입력되는 데이터를 가지고 예측을 한다. 
<img src="https://github.com/junRepository/Predict_CIR/assets/97268718/4596a2c5-58bb-45e3-9613-f7bdbbd8bb8c" width="80%"><br/>

2. ##### 예측하는 코드 작성
    * DNN 모델과 매개변수를 저장한 파일을 불러와 값을 입력하면 예측하는 코드를 작성한다.

    ```py
    def predict_model(data):
        #모델 학습전 실시한 정규화(MinMax-Normalization)을 입력된 데이터에 적용
        loaded_scaler = joblib.load('Normalization.pkl' )
        data_scaled = loaded_scaler.transform(data)
        data_tensor = torch.FloatTensor(data_scaled)

        #DNN 모델 불러오기
        model = NeuralNet(5,2)
        #.pth 형태로 저당된 매개변수를 불러온다.
        model.load_state_dict(torch.load('DNN_model.pth', map_location=torch.device('cpu')))
        #Layer들을 평가 모드로 바꿔준다.
        model.eval()
        
        #torch.no_grad()는 gradient를 계산하지 않게 한다. / backpropagation을 안 하게 한다.
        with torch.no_grad():
            output = model(data_tensor)

        _, predicted_idx = torch.max(output, 1)
        predicted_label = predicted_idx.item()
        return predicted_label
    ``` 
    <br/>

3. ##### FastAPI 코드 작성
###### FastAPI는 Python의 API를 빌드하기 위한 웹 프레임워크이다.
 * GET: 서버상의 값이나 내용을 바꾸지 않고 보여줄 때 사용된다. 
 * POST: 서버상의 값이나 상태를 바꾸기 위해 사용된다. 
 * 리스트(list)형태로 입력받은 데이터들을 예측하여 건전 / 비건전으로 값을 바꿔야하기 때문에 POST를 사용하였다.
 * `DataClassificationRequest`클래스에서 데이터를 검증하고 검증된 데이터를 예측 모델에 전달하고 예측한 결과값을 리턴한다.

    ```py
    @app.get("/")
    def root():
        return "credit classification modle"

    # 입력되는 데이터들이 리스트 형태인지 검증하는 클래스이다. 
    class DataClassificationRequest(BaseModel):
        data: list

    # 검증 클래스에서 검증된 데이터들을 예측 모델에 전달하고
    # 예측한 결과값을 리턴한다.
    @app.post("/predict")
    async def predict(request: DataClassificationRequest):
        list_data = request.data
        prediction = predict_model(list_data)
        return {prediction}
    ```
***

## 2. Docker 와 GCP(Google cloud Platform)를 이용하여 모델 배포
###### [Docker](https://junrepository.github.io/2023/01/30/Docker/): 컨테이너 기술을 이용하여 환경을 구축하고 배포할 수 있게 해주는 플랫폼이다.
###### GCP: Google이 제공하는 클라우드 플랫폼이다. 그중에서 Computer Engine 클라우드 가상서버를 사용하여 배포를 하였다.

<br/>

1. ##### FastAPI 빌드 코드들을 Github을 이용하여 GCP에 보낸다.
<img src="https://github.com/junRepository/Predict_CIR/assets/97268718/9f852215-fc34-48e6-bb23-c36cf3e61eb7" width="80%"><br/>

2. ##### Docker를 이용하여 모델 배포
    * Docker image로 build<br/>
        <img src="https://github.com/junRepository/Predict_CIR/assets/97268718/91433427-36f8-43ee-a0ba-04663a3bd728" width="80%"><br/>
    * Docker image를 Docker container로 배포
        <img src="https://github.com/junRepository/Predict_CIR/assets/97268718/4db4c396-59df-4036-97fd-bd7cba3551f5" width="80%"><br/>

<br/>

## 3. web page를 이용하여 값을 입력하고 예측값을 반환
###### streamlit을 이용하여 web page를 만들었다. web에서 입력한 데이터들이 list 형태로 전달되고, 전달된 데이터를 기반으로 건전 / 비건전 중 하나로 예측하여 다시 web으로 전달이 된다.
###### streamlit : 데이터 시각화를 쉽게 구현하고 간단한 웹서비스를 만들 수 있는 파이썬 라이브러리이다.
###### [web page](https://predictcir.streamlit.app/)   <--- 프로젝트 웹페이지


1. ##### web page 구성
    * 왼쪽 부분에 건전성을 확인하고 싶은 기업의 재무 데이터를 각 항목에 맞게 입력한다.<br/>
    <img src="https://github.com/junRepository/Predict_CIR/assets/97268718/65bd9fe3-8100-4bdd-bbff-8902d6e6c28f" width="60%"><br/>
    * 데이터들을 아래와 같이 입력을 하고 `예측`버튼을 누르면 
    <img src="https://github.com/junRepository/Predict_CIR/assets/97268718/561e1883-4a61-42f8-8289-037aecf24276" width="60%"><br/>
    * 입력된 데이터들이 계산이 되어 list 형태의 5개의 평가 요소로 API에 넘어가는 것을 볼 수 있다.<br/>
    <img src="https://github.com/junRepository/Predict_CIR/assets/97268718/e5dc547c-1059-4eea-8aab-dc842ba7d1f8" width="80%"><br/>
    * 넘어간 데이터를 기반으로 모델이 건전 / 비건전을 예측하여 다시 web page에 전달이 된다. 이진으로 분류하였기 때문에 0은 건전이고 1은 비건전이다.<br/>
    <img src="https://github.com/junRepository/Predict_CIR/assets/97268718/61b5963c-48e5-4547-8750-86e173ab357f" width="80%"><br/>
    * 결과적으로 web page에서 뜨는 화면은 건전으로 출력이 된다.<br/>
    <img src="https://github.com/junRepository/Predict_CIR/assets/97268718/abbf274c-143a-442f-8690-3667373b0c7c" width="60%"><br/>
