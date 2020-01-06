import torch
import torch.nn as nn
import torch.nn.functional as F

# note model = DDC Step Selection + Step Placement
# nove model = Step Placement Score => FCLayer이 곧 Nove의 위치를 나타냄
# DDC Step Selection

#torch.Size([6140, 1, 882])
#torch.Size([6140, 64, 220])
#torch.Size([6140, 128, 73])
#torch.Size([6140, 128, 18])
#torch.Size([6140, 128, 4])
#torch.Size([6140, 512])

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
            nn.AvgPool1d(4))
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(4))
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AvgPool1d(4))
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AvgPool1d(4))  
        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AvgPool1d(4),
            nn.Dropout(0.5))

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 4)

    
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
        
        out = out.view(batch,out.size(1)* out.size(2))

        out = self.fc1(out)
        #print(out.shape)
        out = self.fc2(out)
        #print(out.shape)

        return out
