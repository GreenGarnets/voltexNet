import torch
import torch.nn as nn
import torch.nn.functional as F

# note model = DDC Step Selection + Step Placement
# nove model = Step Placement Score => FCLayer이 곧 Nove의 위치를 나타냄
# DDC Step Selection

#torch.Size([1024, 1, 1764])
#torch.Size([1024, 128, 1764])
#torch.Size([1024, 128, 441])
#torch.Size([1024, 128, 110])
#torch.Size([1024, 256, 27])
#torch.Size([1024, 256, 6])
#torch.Size([1024, 512, 1])
#torch.Size([1024, 512, 1])
#torch.Size([1024, 512])
#torch.Size([1024, 256])
#torch.Size([1024, 4])

class voltexNet(nn.Module):

    def __init__(self):

        super(voltexNet, self).__init__()        

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(3))
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(3))
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AvgPool1d(3))
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AvgPool1d(3))  
        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AvgPool1d(3))
        self.conv7 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AvgPool1d(3))
        self.conv8 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.5))
            
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 4)
        
        self.LSTM = nn.LSTM(input_size = 4, hidden_size = 4, bidirectional=True)

    
    def forward(self, x, batch):

        #print(x.shape)
        out = self.conv1(x)
        #print(out.shape)
        out = self.conv2(out)
        #print(out.shape)
        out = self.conv3(out)
        #print(out.shape)
        out = self.conv4(out)
        #print(out.shape)
        out = self.conv5(out)
        #print(out.shape)
        out = self.conv6(out)
        #print(out.shape)
        out = self.conv7(out)
        #print(out.shape)
        out = self.conv8(out)
        #print(out.shape)
        
        out = out.reshape(batch,out.size(1)* out.size(2))
        #print(out.shape)

        out = self.fc1(out)
        #print(out.shape)
        out = self.fc2(out)
        #print(out.shape)

        #out = out.reshape(1,batch,out.size(1))
        #print(out.shape)
        #out, hidden = self.LSTM(out)
        #print(out.shape)
        #out = out.squeeze()

        return out
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