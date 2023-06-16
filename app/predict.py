import torch
import joblib

class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, output_size): #input: sample의 size  hidden: output의 size
        super(NeuralNet, self).__init__()
        self.input_layer  = torch.nn.Linear(input_size, 8)
        self.hidden_layer1 = torch.nn.Linear(8, 32)
        self.hidden_layer2 = torch.nn.Linear(32, 16)
        self.output_layer = torch.nn.Linear(16, output_size)
        self.dropout = torch.nn.Dropout(0.2)
        self.relu = torch.nn.ReLU()
        self.soft = torch.nn.Softmax(dim=1)
    def forward(self, x):        
        output = self.relu(self.input_layer(x))
        output = self.relu(self.hidden_layer1(output))
        output = self.relu(self.hidden_layer2(output))
        output = self.soft(self.output_layer(output))
        # output = self.output_layer(output)
        return output
    

def predict_model(data):
    model = NeuralNet(7,2)

    loaded_scaler = joblib.load('models/minmax_scaler.pkl' )
    ##AAA
    # data = [[-15.52,-2.87,2.02,-0.63,56.6,76.66,1.13]]
    #D
    # data = [[-4.03,-20.3,-9.41,-10.3,90.02,11.08,0.47]]

    data_scaled = loaded_scaler.transform(data)
    data_tensor = torch.FloatTensor(data_scaled)

    model.load_state_dict(torch.load('models/choose2,eps3000,Acc84.8.pth'))
    model.eval()

    with torch.no_grad():
        output = model(data_tensor)

    _, predicted_idx = torch.max(output, 1)
    predicted_label = predicted_idx.item()

    print("Predicted label:", predicted_label)
    return predicted_label