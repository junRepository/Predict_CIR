import torch
import joblib
from models import NeuralNet

def predict_model(data):

    loaded_scaler = joblib.load('./models/minmax_scaler_feature8.pkl' )
    data_scaled = loaded_scaler.transform(data)
    data_tensor = torch.FloatTensor(data_scaled)
    
    model = NeuralNet(8,2)
    model.load_state_dict(torch.load('./models/predict_model_feature8.pth', map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        output = model(data_tensor)

    _, predicted_idx = torch.max(output, 1)
    predicted_label = predicted_idx.item()
    return predicted_label

model = NeuralNet(8,2)
model.load_state_dict(torch.load('./models/predict_model_feature8.pth', map_location=torch.device('cpu')))
print(torch.load('./models/predict_model_feature8.pth', map_location=torch.device('cpu')))