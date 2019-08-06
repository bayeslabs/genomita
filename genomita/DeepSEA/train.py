import torch
import math
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Load the Drive helper and mount
from google.colab import drive

# This will prompt for authorization.
drive.mount('/content/drive')

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


model = net()

print (model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


labels1 = torch.load('/content/drive/My Drive/colab/dataset/labels_1.pt')

labels2 = torch.load('/content/drive/My Drive/colab/dataset/labels_2.pt')

labels = torch.cat([labels1, labels2])

del labels2

del labels1

seqs1 = torch.load('/content/drive/My Drive/colab/dataset/seqs_1.pt')

seqs2 = torch.load('/content/drive/My Drive/colab/dataset/seqs_2.pt')

seqs = torch.cat([seqs1, seqs2])


del seqs1

del seqs2


def my_loss(output, target):
    nll_loss = 0.0
    
    for i in range(output.shape[0]):
      arr = target[i]*output[i] + (1-target[i])*(1-output[i])
      n_digits = 3
      value = torch.round(arr * 10**n_digits) / (10**n_digits)
      nll_loss = nll_loss + torch.log( value )
      
    nll_loss = (-1.0)*nll_loss
    return nll_loss


import torch.optim as optim
optimizer = optim.SGD(model.parameters(), lr=1, momentum=0.9, weight_decay=1e-6, )


model.train() 
for epoch in range(1): 
  for i in range(10000): 
    inputs = seqs[i] 
    inputs = inputs.float() 
    inputs = inputs.unsqueeze(0).unsqueeze(0) 
    target = labels[i] 
    target = target.float() 
    outputs = model(inputs) 
    loss_ex = my_loss(outputs, target)
    params = model.parameters()
    L2 = 0.0
    i == 0
    L1 = 0.0
    lambda2 = 1e-08
    for p in params:
      i = i+1
      p = p**2
      summ = torch.sum(torch.abs(p))
      L2 = L2 + summ
      
      if i == 8:
        L1 = torch.sum(p)
    
    lambda1 = 5e-07
    loss_ex = loss_ex + lambda1*L2 + lambda2*L1
    print (loss_ex)   
    optimizer.zero_grad() 
    loss_ex.backward() 
    optimizer.step()

print('Finished Training')
