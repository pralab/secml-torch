"""Core code for prototype-based robust training in PyTorch."""
import numpy as np
import torch.nn
import umap
from secmlt.models.base_trainer import BaseTrainer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_auc_score


def cal_prototype(val_features, val_y):
    anchor = []
    for c in range(len(val_y.unique())):
        class_features = val_features[val_y == c]
        prototype = torch.mean(class_features, dim=0)
        anchor.append(prototype)
    anchor = torch.stack(anchor)
    anchor = torch.nn.functional.normalize(anchor, dim=1, p=2)

    return anchor


def get_pseudo_labels(x_features, prototypes):
    similarity = torch.matmul(x_features, prototypes.t())
    pseudo_labels = torch.argmax(similarity, dim=1)

    return pseudo_labels




def calculate_auc(score_benign, gt_poison_labels):
    gt_benign_labels = ~gt_poison_labels
    auc = roc_auc_score(gt_benign_labels, score_benign)
    return auc


def visualization_data(sphere_features_epoch_i_reduced, class_i, gt_poison_epoch_i):
    # if np.unique(gt_poison_epoch_i).shape[0] <2:
    # only draw the plot scale with one color
    # if np.unique(gt_poison_epoch_i).shape[0] ==2 draw the plot scale with two colors
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    if np.unique(gt_poison_epoch_i).shape[0] == 2:
        plt.scatter(sphere_features_epoch_i_reduced[gt_poison_epoch_i == 0, 0],
                    sphere_features_epoch_i_reduced[gt_poison_epoch_i == 0, 1], c='b', label='benign', alpha=0.5)
        plt.scatter(sphere_features_epoch_i_reduced[gt_poison_epoch_i == 1, 0],
                    sphere_features_epoch_i_reduced[gt_poison_epoch_i == 1, 1], c='r', label='poison', alpha=0.5)
        plt.legend()
    else:
        plt.scatter(sphere_features_epoch_i_reduced[:, 0],
                    sphere_features_epoch_i_reduced[:, 1], c='b', alpha=0.5)
    plt.title('Class {}'.format(class_i))
    plt.savefig('class_{}.png'.format(class_i))


def update_weights(dataset, sphere_features_epoch, val_sphere_features, val_features_label, labels_epoch, gt_poison_epoch, indices_epoch, threshold_percent=0.98):
    # convert to numpy
    sphere_features_epoch = sphere_features_epoch.numpy()
    labels_epoch = labels_epoch.numpy()
    indices_epoch = indices_epoch.numpy()
    gt_poison_epoch = gt_poison_epoch.numpy()

    # umap dimension reduction
    umap_model_10d = umap.UMAP(n_components=2, n_jobs=-1, metric='cosine')  # , random_state=42)
    pdf_all_classes, weight_indices_l, gt_poison_flag_l = [], [], []
    for class_i in np.unique(labels_epoch):
        val_sphere_features_i = val_sphere_features[val_features_label == class_i]  # Anchor points
        sphere_features_epoch_i = sphere_features_epoch[labels_epoch == class_i]  # Untrusted samples
        indices_epoch_i = indices_epoch[labels_epoch == class_i]
        gt_poison_epoch_i = gt_poison_epoch[labels_epoch == class_i]

        concatenated_data = np.concatenate((val_sphere_features_i, sphere_features_epoch_i), axis=0)
        reduced_data_10d = umap_model_10d.fit_transform(concatenated_data)
        val_sphere_features_i_reduced, sphere_features_epoch_i_reduced = reduced_data_10d[:len(val_sphere_features_i)],\
            reduced_data_10d[len(val_sphere_features_i):]

        # Step 2: Calculate the PDF values
        pdf_class_i = []
        for val_sphere_features_i_j in val_sphere_features_i_reduced:
            mvn = multivariate_normal(mean=val_sphere_features_i_j, cov=np.eye(len(val_sphere_features_i_j)))
            log_prob = mvn.logpdf(sphere_features_epoch_i_reduced)
            pdf = np.exp(log_prob)
            pdf_class_i.append(pdf)
        pdf_class_i = np.stack(pdf_class_i, axis=1)
        pdf_class_i = np.max(pdf_class_i, axis=1)
        pdf_all_classes.append(pdf_class_i)
        weight_indices_l.append(indices_epoch_i)
        gt_poison_flag_l.append(gt_poison_epoch_i)
        # vislualization the data distribution
        visualization_data(sphere_features_epoch_i_reduced, class_i, gt_poison_epoch_i)

    pdf_all_classes = np.concatenate(pdf_all_classes)
    weight_indices_l = np.concatenate(weight_indices_l)
    gt_poison_epochs = np.concatenate(gt_poison_flag_l)

    # Normalize across all classes
    # get threshold based on the threshold_percent
    sorted_pdf = np.sort(pdf_all_classes)[::-1]
    top_percent = int(threshold_percent * sorted_pdf.size)  # - 1  #  normalize pdf lower than threshold P(pdf<index_80_percent)=0.9 others set as 1
    threshold = sorted_pdf[top_percent]

    # Create mask for values below the cutoff and normalize
    bottom_half_mask = pdf_all_classes <= threshold
    bottom_half_values = pdf_all_classes[bottom_half_mask]
    epsilon = 1e-10  # small constant to avoid division by zero
    bottom_half_normalized = 2 * ((bottom_half_values - np.min(bottom_half_values)) / (
            np.max(bottom_half_values) - np.min(bottom_half_values) + epsilon)) - 1


    # Assign 1 to the top 20% and normalize the bottom 80%
    pdf_all_classes[~bottom_half_mask] = 1
    pdf_all_classes[bottom_half_mask] = bottom_half_normalized

    # Update weights with momentum
    momentum_alpha = 0.5
    # pdf_all_classes sorted according to the indices
    indices_sorted = np.argsort(weight_indices_l)
    pdf_all_classes_sorted = pdf_all_classes[indices_sorted]
    gt_poison_flag_l_sorted = gt_poison_epochs[indices_sorted]

    # pdf_all_classes_sorted to tensor
    dataset.weights = momentum_alpha * dataset.weights + (1 - momentum_alpha) * torch.tensor(pdf_all_classes_sorted, dtype=torch.float32).to(dataset.weights.device)
    auc = calculate_auc(dataset.weights, gt_poison_flag_l_sorted)
    # print('auc: {:.3f}'.format(auc))

def calculate_tpr_fpr(pred, ground_truth):
    true_positives = np.sum((pred == 1) & (ground_truth == 1))
    false_positives = np.sum((pred == 1) & (ground_truth == 0))
    false_negatives = np.sum((pred == 0) & (ground_truth == 1))
    true_negatives = np.sum((pred == 0) & (ground_truth == 0))
    tpr = true_positives / (true_positives + false_negatives + 1e-10)
    fpr = false_positives / (false_positives + true_negatives + 1e-10)
    return tpr, fpr


class RobustPyTorchTrainer(BaseTrainer):
    """Trainer for PyTorch models."""

    def __init__(
        self,
        validation_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs: int = 5,
        loss: torch.nn.Module = None,
        scheduler: _LRScheduler = None,
    ) -> None:
        """
        Create PyTorch trainer.

        Parameters
        ----------
        validation_dataloader : DataLoader
            DataLoader for prototype and feature distance estimation
        optimizer : torch.optim.Optimizer
            Optimizer to use for training the model.
        epochs : int, optional
            Number of epochs, by default 5.
        loss : torch.nn.Module, optional
            Loss to minimize, by default None.
        scheduler : _LRScheduler, optional
            Scheduler for the optimizer, by default None.
        """
        self._validation_dataloader = validation_dataloader
        self._epochs = epochs
        self._optimizer = optimizer
        self._loss = loss if loss is not None else torch.nn.CrossEntropyLoss()
        self._scheduler = scheduler

    def train(self, model: torch.nn.Module, dataloader: DataLoader) -> torch.nn.Module:
        """
        Train model with given loader.

        Parameters
        ----------
        model : torch.nn.Module
            Pytorch model to be trained.
        dataloader : DataLoader
            Train data loader.

        Returns
        -------
        torch.nn.Module
            Trained model.
        """
        device = next(model.parameters()).device
        model = model.train()
        for e_id in range(self._epochs):
            sphere_features_l, labels_l, indices_l = [], [], []
            val_sphere_features, val_features_label = None, None
            pred_poison_l, gt_poison_l = [], []
            for _, batch in enumerate(dataloader):
                x, y, w, i = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
                batch_val = self.get_batch_val(self._validation_dataloader)
                val_x, val_y = batch_val[0].to(device), batch_val[1].to(device)
                combined_x = torch.cat([x, val_x], dim=0)
                self._optimizer.zero_grad()
                combined_outputs = model(combined_x)
                outputs = combined_outputs[:len(x)]
                sphere_features = model.return_sphere_space()
                sphere_features_l.append(sphere_features[:len(x)].detach().cpu())
                val_sphere_features = sphere_features[len(x):].detach().cpu()
                val_features_label = val_y.detach().cpu()
                indices_l.append(i.detach().cpu())
                labels_l.append(y.detach().cpu())
                # generate prototypes
                prototypes = cal_prototype(sphere_features[len(x):], val_y)
                # get psudeo labels based on closest prototype
                pseudo_labels = get_pseudo_labels(sphere_features[:len(x)], prototypes)
                consistency_flag = y == pseudo_labels
                pred_poison_l.append((~consistency_flag).long().detach().cpu())
                gt_poison_l.append(batch[4].long().detach().cpu())
                # compute the Tpr and fpr between consistency and poison_flags batch[4]
                loss = torch.mean(self._loss(outputs[consistency_flag], y[consistency_flag]) * w[consistency_flag].to(device))
                loss.backward()
                self._optimizer.step()
            if self._scheduler is not None:
                self._scheduler.step()
            pred_poison_l = torch.cat(pred_poison_l, dim=0)
            gt_poison_l = torch.cat(gt_poison_l, dim=0)
            tpr, fpr = calculate_tpr_fpr(pred_poison_l.cpu().numpy(), gt_poison_l.cpu().numpy())
            # print('tpr, fpr: {}, {}'.format(tpr, fpr))
            if e_id % 2 == 0:
                # update the weights based on the features
                sphere_features_epoch = torch.cat(sphere_features_l, dim=0)
                labels_epoch = torch.cat(labels_l, dim=0)
                indices_epoch = torch.cat(indices_l, dim=0)
                # update weights in dataloader
                update_weights(dataloader.dataset, sphere_features_epoch, val_sphere_features, val_features_label, labels_epoch, gt_poison_l, indices_epoch)

        return model

    def get_batch_val(self, val_dl):
        # supervised learning
        try:
            batch_val = next(iter(val_dl))
        except StopIteration:
            val_dl = DataLoader(val_dl.dataset, batch_size=len(val_dl.dataset), shuffle=True,
                                num_workers=val_dl.num_workers)
            batch_val = next(iter(val_dl))
        while len(set(batch_val[1].tolist())) != len(
                val_dl.dataset.dataset.dataset.classes):  # ensure there is at least one element for each class
            batch_val = next(iter(val_dl))

        return batch_val