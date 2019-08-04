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
    print (x.shape)
    x = self.conv1(x)
    print (x.shape)
    x = F.max_pool2d(x, (4, 1), stride=4)
    print (x.shape)
    x = self.conv2(x)
    print (x.shape)
    x = F.max_pool2d(x, (4, 1), stride=4)
    print (x.shape)
    x = self.conv3(x)
    print (x.shape)
    x = torch.flatten(x)
    print (x.shape)
    x = torch.sigmoid(self.fc1(x))
    print (x.shape)
    return x


model = net()
# print(model)
# print (list(net.parameters())[0])  

# Initialize optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

# Load the Drive helper and mount
from google.colab import drive

# This will prompt for authorization.
drive.mount('/content/drive')

print (model)

input = torch.randn(1, 1, 1000, 4)
out = model(input)

out.shape
