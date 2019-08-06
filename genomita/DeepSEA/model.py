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
