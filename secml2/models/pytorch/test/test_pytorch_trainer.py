from torch.optim import Adam

from secml2.models.pytorch.base_pytorch_trainer import BasePyTorchTrainer
from secml2.models.pytorch.test.base_pytorch import BasePytorchTests


class TestPytorchTrainer(BasePytorchTests):
    def test_train_net(self):
        old_weights = [t.clone() for t in list(self._net.parameters())]
        trainer = BasePyTorchTrainer(
            optimizer=Adam(lr=1e-3, params=self._net.parameters())
        )
        fit_net = trainer.train(self._net, self._dataloader)

        sum_old_weights = sum([t.sum().item() for t in old_weights])
        sum_new_weights = sum([t.sum().item() for t in fit_net.parameters()])

        self.assertNotEqual(sum_old_weights, sum_new_weights)

    def test_test_accuracy(self):
        # TODO: implement test with mock network
        raise NotImplementedError("test yet to be implemented")
