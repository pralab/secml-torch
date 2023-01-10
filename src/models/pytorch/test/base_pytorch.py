import unittest
from functools import reduce

import torch.nn
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor


class Net(Module):
	def __init__(self, input_size, output_size):
		super(Net, self).__init__()
		self._l1 = torch.nn.Linear(reduce(lambda x, y: x * y, input_size), out_features=output_size)

	def forward(self, x: torch.Tensor):
		return self._l1(x.view(x.shape[0], -1))


class BasePytorchTests(unittest.TestCase):
	def setUp(self):
		self.input_shape = (3, 224, 224)
		self.output_shape = 2
		self._data = FakeData(size=10, num_classes=self.output_shape, image_size=self.input_shape, transform=ToTensor())
		self._dataloader = DataLoader(self._data)
		self._net = Net(self.input_shape, self.output_shape)
