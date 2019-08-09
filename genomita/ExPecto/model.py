class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
		self.sp1_s = nn.Conv1d(in_channels = 320, out_channels = 320, kernel_size = (4, 1,  8))

		self.sp2_s = nn.Conv1d(in_channels = 320, out_channels = 320, kernel_size = (320, 1, 8))

		self.sp3_s = nn.Conv1d(in_channels = 320, out_channels = 480, kernel_size = (320, 1, 8))
        
		self.sp4_s = nn.Conv1d(in_channels = 480, out_channels = 480, kernel_size = (480, 1, 8))
        
		self.sp5_s = nn.Conv1d(in_channels = 480, out_channels = 640, kernel_size = (480, 1, 8))
        
		self.sp6_s = nn.Conv1d(in_channels = 640, out_channels = 640, kernel_size = (640, 1, 8))

		self.fc1 = nn.Linear(67840, 2003)  
		self.fc2 = nn.Linear(2003, 2002)

	def forward(self, x):
		x = F.relu(self.sp1_s(x))
		x = F.relu(self.sp2_s(x))
		x = F.max_pool1d(F.dropout(x), (4, 1))
		x = F.relu(self.sp3_s(x))
		x = F.relu(self.sp4_s(x))
		x = F.max_pool1d(F.dropout(x, p = 0.2), (4, 1))
		x = F.relu(self.sp5_s(x))
		x = F.relu(self.sp6_s(x))
		x = F.dropout(x, p = 0.2)
		x = F.sigmoid(self.fc2(x))
		return x
