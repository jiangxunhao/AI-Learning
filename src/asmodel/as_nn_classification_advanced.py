import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from util import P4util


class ASClassificationAdvancedDataset(Dataset):
    def __init__(self, features, labels, performances):
        self.features = features
        self.labels = labels
        self.performances = performances

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx, :], self.labels[idx], self.performances[idx]


class ASClassificationAdvanced(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 150),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.ReLU(),
            nn.Linear(150, output_dim),
            nn.Softmax(1)
        )

    def forward(self, X):
        digits = self.net(X)
        return digits


class RegretLoss(nn.Module):
    def __init__(self):
        super(RegretLoss, self).__init__()

    def forward(self, pred, labels, performances):
        pred_performances = torch.mul(performances, pred)
        pred_performance = torch.sum(pred_performances, dim=1, keepdim=True)
        losses = torch.add(pred_performance, torch.neg(performances[torch.arange(len(labels)), labels]))
        loss_value = torch.mean(losses)
        return loss_value


def as_classification_advanced_train(features_train, performances_train, save_path, lr=1e-4, epochs=1500, batch_size=64):
    labels_train = torch.argmin(performances_train, dim=1)
    train_dataset = ASClassificationAdvancedDataset(features_train, labels_train, performances_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = ASClassificationAdvanced(features_train.shape[1], performances_train.shape[1])
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    n_samples = features_train.shape[0]
    loss_function = RegretLoss()

    for ep_id in range(epochs):
        for X, y, p in train_dataloader:
            pred = model(X)

            loss = loss_function(pred, y, p)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        pred = model(features_train)
        pred_labels = pred.argmax(dim=1)
        n_corrects = (pred_labels == labels_train).sum()
        # total_loss = loss_function(cost(pred_labels, performances_train), cost(labels_train, performances_train))
        total_loss = loss_function(pred, labels_train, performances_train)
        accuracy = n_corrects / n_samples
        gap = P4util.sbs_vbs_gap(pred_labels, performances_train)

        print(f"epoch: {ep_id},\t total_loss: {total_loss},\t accuracy: {accuracy},\t sbs_vbs_gap: {gap}")

    torch.save(model.state_dict(), save_path)
