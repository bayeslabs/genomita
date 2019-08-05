import h5py
import os
import torch
import numpy as np

# Load the Drive helper and mount
from google.colab import drive

# This will prompt for authorization.
drive.mount('/content/drive')

h5f = h5py.File(os.path.join('/content/drive/My Drive/colab/data', 'deepsea_train/train.mat'), 'r')

train_seqs = h5f['trainxdata']
# train_labels = h5f['traindata']

num_examples = 500000
data_seqs = []

for i in range(num_examples):
  data_seqs.append(h5f['trainxdata'][:, :, i])
  
print (i)
data_seqs = np.array(data_seqs)
data_seqs

data_seqs = torch.tensor(data_seqs)
torch.save(data_seqs, '/content/drive/My Drive/colab/dataset/seqs_1.pt')
data_seqs = []

for i in range(num_examples, 2*num_examples):
  data_seqs.append(h5f['trainxdata'][:, :, i])
  
print (i)
data_seqs = np.array(data_seqs)
data_seqs = torch.tensor(data_seqs)

torch.save(data_seqs, '/content/drive/My Drive/colab/dataset/seqs_2.pt')

data_seqs = []
for i in range(2*num_examples, 3*num_examples):
  data_seqs.append(h5f['trainxdata'][:, :, i])
  
print (i)
data_seqs = np.array(data_seqs)
data_seqs = torch.tensor(data_seqs)
torch.save(data_seqs, '/content/drive/My Drive/colab/dataset/seqs_3.pt')

data_seqs = []
for i in range(3*num_examples, 4*num_examples):
  data_seqs.append(h5f['trainxdata'][:, :, i])
  
print (i)
data_seqs = np.array(data_seqs)
data_seqs = torch.tensor(data_seqs)
torch.save(data_seqs, '/content/drive/My Drive/colab/dataset/seqs_4.pt')

data_seqs = []
for i in range(4*num_examples, 5*num_examples):
  data_seqs.append(h5f['trainxdata'][:, :, i])
  
print (i)
data_seqs = np.array(data_seqs)
data_seqs = torch.tensor(data_seqs)
torch.save(data_seqs, '/content/drive/My Drive/colab/dataset/seqs_5.pt')

data_seqs = []
for i in range(5*num_examples, 6*num_examples):
  data_seqs.append(h5f['trainxdata'][:, :, i])
  
print (i)
data_seqs = np.array(data_seqs)
data_seqs = torch.tensor(data_seqs)
torch.save(data_seqs, '/content/drive/My Drive/colab/dataset/seqs_6.pt')

data_seqs = []
for i in range(6*num_examples, 7*num_examples):
  data_seqs.append(h5f['trainxdata'][:, :, i])
  
print (i)
data_seqs = np.array(data_seqs)
data_seqs = torch.tensor(data_seqs)
torch.save(data_seqs, '/content/drive/My Drive/colab/dataset/seqs_7.pt')

data_seqs = []
for i in range(7*num_examples, 8*num_examples):
  data_seqs.append(h5f['trainxdata'][:, :, i])
  
print (i)
data_seqs = np.array(data_seqs)
data_seqs = torch.tensor(data_seqs)
torch.save(data_seqs, '/content/drive/My Drive/colab/dataset/seqs_8.pt')

data_seqs = []
for i in range(8*num_examples, 4400000): 
  data_seqs.append(h5f['trainxdata'][:, :, i])
  
print (i)
data_seqs = np.array(data_seqs)
data_seqs = torch.tensor(data_seqs)
torch.save(data_seqs, '/content/drive/My Drive/colab/dataset/seqs_9.pt')
  
