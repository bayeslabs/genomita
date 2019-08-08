import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import model

#if you want to import from google colab folder you can use this below commented code 
# from google.colab import drive
# drive.mount('/content/drive')

model = net()
model.load_state_dict(torch.load('/data/trained_model.pth'))

seqs = torch.load('/data/seqs_9.pt')

model.eval()
for i in range(seqs.shape[0]):
  inputs = seqs[i] 
  inputs = inputs.float() 
  inputs = inputs.unsqueeze(0).unsqueeze(0)
  prediction = model(inputs)
