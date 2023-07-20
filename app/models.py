import torch

class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, output_size): #input: sample의 size  hidden: output의 size
        super(NeuralNet, self).__init__()
        self.input_layer  = torch.nn.Linear(input_size, 8)
        self.batchnorm = torch.nn.BatchNorm1d(8)
        self.hidden_layer1 = torch.nn.Linear(8, 32)
        self.batchnorm1 = torch.nn.BatchNorm1d(32)
        self.hidden_layer2 = torch.nn.Linear(32, 518)
        self.batchnorm2 = torch.nn.BatchNorm1d(518)
        self.hidden_layer3 = torch.nn.Linear(518, 518)
        self.batchnorm3 = torch.nn.BatchNorm1d(518)
        self.hidden_layer5 = torch.nn.Linear(518, 32)
        self.batchnorm5 = torch.nn.BatchNorm1d(32)
        self.hidden_layer6 = torch.nn.Linear(32, 16)
        self.batchnorm6 = torch.nn.BatchNorm1d(16)
        self.output_layer = torch.nn.Linear(16, output_size)
        self.dropout = torch.nn.Dropout(0.2)
        self.relu = torch.nn.LeakyReLU()
        self.soft = torch.nn.Softmax()
    def forward(self, x):        
        output = self.batchnorm(self.input_layer(x))
        output = self.dropout(self.relu(self.batchnorm1(self.hidden_layer1(output))))
        output = self.dropout(self.relu(self.batchnorm2(self.hidden_layer2(output))))
        output = self.dropout(self.relu(self.batchnorm3(self.hidden_layer3(output))))
        output = self.dropout(self.relu(self.batchnorm3(self.hidden_layer3(output))))
        output = self.dropout(self.relu(self.batchnorm3(self.hidden_layer3(output))))
        output = self.dropout(self.relu(self.batchnorm3(self.hidden_layer3(output))))
        output = self.dropout(self.relu(self.batchnorm3(self.hidden_layer3(output))))
        output = self.dropout(self.relu(self.batchnorm3(self.hidden_layer3(output))))
        output = self.dropout(self.relu(self.batchnorm3(self.hidden_layer3(output))))
        output = self.dropout(self.relu(self.batchnorm3(self.hidden_layer3(output))))
        output = self.dropout(self.relu(self.batchnorm5(self.hidden_layer5(output))))
        output = self.dropout(self.relu(self.batchnorm6(self.hidden_layer6(output))))
        output = self.output_layer(output)
        return output

