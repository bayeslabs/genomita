import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class net (nn.Module):
  def __init__(self):
    super(net, self).__init__()
    self.conv1 = nn.Conv2d(1, 320 , (8, 4))
    self.dp1 = nn.Dropout(p = 0.2)
    self.conv2 = nn.Conv2d(320, 480, (8, 1))
    self.dp2 = nn.Dropout(p = 0.2)
    self.conv3 = nn.Conv2d(480, 960, (8, 1))
    self.dp3 = nn.Dropout(p = 0.5)
    width  = 1000
    nchannel = math.floor((math.floor((width-7)/4.0)-7)/4.0)-7
    print (nchannel)
    self.fc1 = nn.Linear( 960*nchannel, 919)
   
  def forward(self, x):
    x = self.conv1(x)
    x = F.max_pool2d(x, (4, 1), stride=4)
    x = self.dp1(x)
    x = self.conv2(x)
    x = F.max_pool2d(x, (4, 1), stride=4)
    x = self.dp2(x)
    x = self.conv3(x)
    x = self.dp3(x)
    x = torch.flatten(x)
    x = self.fc1(x)
    x = torch.sigmoid(x)
    return x

# Load the Drive helper and mount
from google.colab import drive

# This will prompt for authorization.
drive.mount('/content/drive')

model = net()
model.load_state_dict(torch.load('/content/drive/My Drive/colab/dataset/trained_model.pth'))

seqs = torch.load('/content/drive/My Drive/colab/dataset/seqs_9.pt')

model.eval()
for i in range(seqs.shape[0]):
  inputs = seqs[i] 
  inputs = inputs.float() 
  inputs = inputs.unsqueeze(0).unsqueeze(0)
  prediction = model(inputs)
