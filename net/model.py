import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class voltexNet(nn.Module):

    def __init__(self):

        super(voltexNet, self).__init__()        

        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(3,stride=3))
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AvgPool1d(3,stride=3))
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AvgPool1d(3,stride=3))
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AvgPool1d(4,stride=4))
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AvgPool1d(4,stride=4))

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

        #self.lstm = nn.LSTM()
        
        #self.LSTM = nn.LSTM(input_size = 4, hidden_size = 2, bidirectional=True)
        #self.fc3 = nn.Linear(hidden_size*2,output_size)
        #self.output = F.softmax(linear(output),1)

    
    def forward(self, x):
        x = x.squeeze()

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        #print(out.shape)
        
        out = out.reshape(out.size(0), out.size(1)* out.size(2))
        #print(out.shape)

        out = self.fc1(out)
        out = self.fc2(out)
        #out = self.fc2(out)

        # out = out.reshape(1,batch,out.size(1))
        # out, hidden = self.LSTM(out)
        # out = out.squeeze()

        return torch.sigmoid(self.fc3(out))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model = voltexNet().to(device)

    summary(model, [(3, 883, 1)])

'''
class VoltexLSTM

    def __init__(self):

        super(voltexNet, self).__init__()       
        self.LSTM = nn.LSTM(input_size = 4, hidden_size = 4,batch_first = True) 

    def forward(self, x, batch):

        print(x.shape)
        out, hidden = self.LSTM(out)
        print(out.shape)
'''