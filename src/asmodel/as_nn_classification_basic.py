from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch


from util import P4util


class ASClassificationBasicDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx, :], self.labels[idx]


class ASClassificationBasic(nn.Module):
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


def as_classification_basic_train(features_train, performances_train, save_path, lr=1e-3, epochs=1000, batch_size=64):
    labels_train = torch.argmin(performances_train, dim=1)
    train_dataset = ASClassificationBasicDataset(features_train, labels_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = ASClassificationBasic(features_train.shape[1], performances_train.shape[1])
    loss_function = nn.functional.cross_entropy
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    n_samples = features_train.shape[0]


    for ep_id in range(epochs):
        for X, y in train_dataloader:
            pred = model(X)
            # print(pred)
            # print(y)
            loss = loss_function(pred, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        pred = model(features_train)
        pred_labels = pred.argmax(dim=1)
        n_corrects = (pred_labels == labels_train).sum()
        total_loss = loss_function(pred, labels_train, reduction="mean")
        accuracy = n_corrects / n_samples
        gap = P4util.sbs_vbs_gap(pred_labels, performances_train)


        print(f"epoch: {ep_id},\t total_loss: {total_loss},\t accuracy: {accuracy},\t sbs_vbs_gap: {gap}")


    torch.save(model.state_dict(), save_path)
