import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from util import P4util


class ASRegressionDataset(Dataset):
    def __init__(self, features, performances):
        self.features = features
        self.performances = performances

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx], self.performances[idx]


class ASRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 150),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.ReLU(),
            nn.Linear(150, output_dim)
        )

    def forward(self, X):
        digits = self.net(X)
        return digits


def as_regression_train(features_train, performances_train, save_path, lr=1e-3, epochs=1000, batch_size=64):
    train_dataset = ASRegressionDataset(features_train, performances_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = ASRegression(features_train.shape[1], performances_train.shape[1])
    loss_function = nn.functional.mse_loss
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    n_samples = features_train.shape[0]
    true_chosen = torch.argmin(performances_train, dim=1)

    for ep_id in range(epochs):

        for X, y in train_dataloader:
            pred = model(X)

            loss = loss_function(pred, y, reduction="mean")
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        pred_performances = model(features_train)
        pred_chosen = torch.argmin(pred_performances, dim=1)
        n_corrects = (true_chosen == pred_chosen).sum()
        mean_loss = loss_function(pred_performances, performances_train, reduction="mean") / n_samples

        accuracy = n_corrects / n_samples

        gap = P4util.sbs_vbs_gap(pred_chosen, performances_train)

        print(f"epoch: {ep_id},\t avg_loss: {mean_loss},\t accuracy: {accuracy},\t sbs_vbs_gap: {gap}")

    torch.save(model.state_dict(), save_path)


