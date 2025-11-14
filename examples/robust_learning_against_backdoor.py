import torch, torchvision

from models.mnist_net import MNISTNet_with_sphere_space
from secmlt.adv.poisoning.backdoor import BackdoorDatasetPyTorch, BackdoorDatasetPyTorchWithCoverSample
from secmlt.metrics.classification import Accuracy, AttackSuccessRate
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier
from torch.utils.data import DataLoader, Subset

from secmlt.models.pytorch.robust_trainer import RobustPyTorchTrainer


def get_mnist_dataloaders(batch_size, target_label, portion, cover_portion, dataset_path, num_workers):
    def apply_patch(x: torch.Tensor) -> torch.Tensor:
        x[:, 0, 24:28, 24:28] = 1.0
        return x

    training_dataset = torchvision.datasets.MNIST(
        transform=torchvision.transforms.ToTensor(),
        train=True,
        root=dataset_path,
        download=True,
    )
    tr_ds = BackdoorDatasetPyTorchWithCoverSample(
        training_dataset,
        data_manipulation_func=apply_patch,
        trigger_label=target_label,
        portion=portion,
        cover_portion=cover_portion,
    )

    ts_ds = torchvision.datasets.MNIST(
        transform=torchvision.transforms.ToTensor(),
        train=False,
        root=dataset_path,
        download=True,
    )
    # filter out the samples with target label in test dataset
    filtered_indices = [i for i, (_, label) in enumerate(ts_ds) if label != target_label]
    ts_ds_non_target = Subset(ts_ds, filtered_indices)
    p_ts_ds = BackdoorDatasetPyTorch(ts_ds_non_target, data_manipulation_func=apply_patch)

    tr_dl = DataLoader(tr_ds, batch_size, shuffle=True, num_workers=num_workers)
    ts_dl = DataLoader(ts_ds, batch_size, shuffle=False, num_workers=num_workers)
    p_ts_dl = DataLoader(p_ts_ds, batch_size, shuffle=False, num_workers=num_workers)

    return tr_dl, ts_dl, p_ts_dl


def get_validation_data(tr_ds, num_sample, num_workers):
    subset_indics = []
    for class_i in range(len(tr_ds.dataset.classes)):
        cnt = 0
        for i, (x, label) in enumerate(tr_ds.dataset):
            if label == class_i:
                if i not in tr_ds.poisoned_indexes and i not in tr_ds.cover_indexes: # labeled dara from benign samples
                    subset_indics.append(i)
                    cnt += 1
                else:
                    pass

            if cnt == num_sample:
                break

    subset_indics_left = []
    for i in range(len(tr_ds)):
        if i in subset_indics:
            continue
        else:
            subset_indics_left.append(i)

    val_ds = Subset(tr_ds, subset_indics)
    val_dl = DataLoader(dataset=val_ds, batch_size=len(val_ds), shuffle=True, num_workers=num_workers)

    return val_dl, subset_indics, subset_indics_left


def evaluate_model(model, ts_dl, p_ts_dl, target_label):
    # test accuracy without backdoor
    acc = Accuracy()(model, ts_dl)
    print("acc: {:.3f}".format(acc.item()))

    asr = AttackSuccessRate(y_target=target_label)(model, p_ts_dl)
    print("asr: {:.3f}".format(asr.item()))

    return acc, asr


def main():
    device = "cuda:0"
    batch_size, target_label, portion, cover_portion, dataset_path, num_workers = 1024, 1, 0.01, 0.01, "example_data/datasets/", 0
    # training, test, and p_test dataset
    tr_dl, ts_dl, p_ts_dl = get_mnist_dataloaders(batch_size, target_label, portion, cover_portion, dataset_path, num_workers)
    # validation dataloader
    val_dl, _, _ = get_validation_data(tr_dl.dataset, 50, num_workers)
    # define model, optimizer
    model = MNISTNet_with_sphere_space()
    model.to(device)
    # define robust trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = torch.nn.CrossEntropyLoss(reduction="none")
    trainer = RobustPyTorchTrainer(validation_dataloader=val_dl, optimizer=optimizer, epochs=10, loss=loss)
    # train the model with pgrl robust training and test the performance
    model = trainer.train(model, tr_dl)
    model = BasePytorchClassifier(model)

    acc, asr = evaluate_model(model, ts_dl, p_ts_dl, target_label)


if __name__ == "__main__":
    main()