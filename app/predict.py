import torch
import joblib

# class NeuralNet(torch.nn.Module):
#     def __init__(self, input_size, output_size): #input: sample의 size  hidden: output의 size
#         super(NeuralNet, self).__init__()
#         self.input_layer  = torch.nn.Linear(input_size, 8)
#         self.hidden_layer1 = torch.nn.Linear(8, 32)
#         self.hidden_layer2 = torch.nn.Linear(32, 16)
#         self.output_layer = torch.nn.Linear(16, output_size)
#         self.dropout = torch.nn.Dropout(0.2)
#         self.relu = torch.nn.ReLU()
#         self.soft = torch.nn.Softmax(dim=1)
#     def forward(self, x):        
#         output = self.relu(self.input_layer(x))
#         output = self.relu(self.hidden_layer1(output))
#         output = self.relu(self.hidden_layer2(output))
#         output = self.soft(self.output_layer(output))
#         # output = self.output_layer(output)
#         return output
    

class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, output_size): #input: sample의 size  hidden: output의 size
        super(NeuralNet, self).__init__()
        self.input_layer  = torch.nn.Linear(input_size, 8)
        self.batchnorm = torch.nn.BatchNorm1d(8)
        self.hidden_layer1 = torch.nn.Linear(8, 32)
        self.batchnorm1 = torch.nn.BatchNorm1d(32)
        self.hidden_layer2 = torch.nn.Linear(32, 256)
        self.batchnorm2 = torch.nn.BatchNorm1d(256)
        self.hidden_layer3 = torch.nn.Linear(256, 128)
        self.batchnorm3 = torch.nn.BatchNorm1d(128)
        self.hidden_layer4 = torch.nn.Linear(128, 64)
        self.batchnorm4 = torch.nn.BatchNorm1d(64)
        self.hidden_layer5 = torch.nn.Linear(64, 16)
        self.batchnorm5 = torch.nn.BatchNorm1d(16)
        self.output_layer = torch.nn.Linear(16, output_size)
        self.dropout = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()
        self.soft = torch.nn.Softmax(dim=1)
    def forward(self, x):        
        output = self.relu(self.batchnorm(self.input_layer(x)))
        output = self.relu(self.batchnorm1(self.hidden_layer1(output)))
        output = self.dropout(output)
        output = self.relu(self.batchnorm2(self.hidden_layer2(output)))
        output = self.dropout(output)
        output = self.relu(self.batchnorm3(self.hidden_layer3(output)))
        output = self.dropout(output)
        output = self.relu(self.batchnorm4(self.hidden_layer4(output)))
        output = self.dropout(output)
        output = self.relu(self.batchnorm5(self.hidden_layer5(output)))
        output = self.dropout(output)
        # output = self.soft(self.output_layer(output))
        output = self.output_layer(output)
        return output
    

def predict_model(data):
    model = NeuralNet(8,2)

    loaded_scaler = joblib.load('/models/minmax_scaler_feature8.pkl' )
    data_scaled = loaded_scaler.transform(data)
    data_tensor = torch.FloatTensor(data_scaled)

    model.load_state_dict(torch.load('/models/predict_model_feature8.pth', map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        output = model(data_tensor)

    _, predicted_idx = torch.max(output, 1)
    predicted_label = predicted_idx.item()

    print("Predicted label:", predicted_label)
    return predicted_label

data = [[28.35,8.76,19.83,5.22,5.56,50.35,98.6,0.75]]
predict_model(data)