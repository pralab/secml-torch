import torch
from torch.optim import Adam

from src.models.pytorch.base_pytorch_nn import BasePytorchClassifier
from src.models.pytorch.base_pytorch_trainer import BasePyTorchTrainer
from src.models.pytorch.test.base_pytorch import BasePytorchTests


class TestPytorchNN(BasePytorchTests):
    def setUp(self):
        super(TestPytorchNN, self).setUp()
        self._trainer = BasePyTorchTrainer(
            optimizer=Adam(lr=1e-3, params=self._net.parameters()),
            loss=torch.nn.CrossEntropyLoss(),
        )

    def test_fit_raise_exception_no_trainer(self):
        test_net = BasePytorchClassifier(self._net, trainer=None)
        with self.assertRaises(ValueError):
            test_net.train(self._dataloader)

    def test_fit_ok(self):
        test_net = BasePytorchClassifier(self._net, trainer=self._trainer)
        test_net.train(self._dataloader)
        self.assertTrue(test_net)
