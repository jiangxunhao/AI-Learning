import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class ASBinaryClassificationDataset(Dataset):
    def __init__(self, features, labels, performances):
        self.features = features
        self.labels = labels
        self.performances = performances

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx, :], self.labels[idx], self.performances[idx]


class ASBinaryClassification(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 150),
            nn.Sigmoid(),
            nn.Linear(150, 150),
            nn.Sigmoid(),
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


def as_binary_classification_train(features_train, performances_train, save_path, lr=1e-4, epochs=1000, batch_size=64):
    for i in range(performances_train.shape[1]):
        for j in range(i+1, performances_train.shape[1]):

            labels_train = torch.argmin(performances_train[:, (i, j)], dim=1)
            binary_performances_train = performances_train[:, (i, j)]
            train_dataset = ASBinaryClassificationDataset(features_train, labels_train, binary_performances_train)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            model = ASBinaryClassification(features_train.shape[1], 2)
            loss_function = RegretLoss()
            optimiser = torch.optim.Adam(model.parameters(), lr=lr)

            n_samples = features_train.shape[0]

            for ep_id in range(epochs):
                for X, y, p in train_dataloader:
                    pred = model(X)
                    # print(pred)
                    # print(y)
                    loss = loss_function(pred, y, p)
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

                pred = model(features_train)
                pred_labels = pred.argmax(dim=1)
                n_corrects = (pred_labels == labels_train).sum()
                loss = loss_function(pred, labels_train, binary_performances_train)

                print(f"first_algorithm: {i},\t second_algorithm: {j},\t epoch: {ep_id},"
                      f"\t avg_loss: {loss / n_samples},\t accuracy: {n_corrects} / {n_samples}")

            torch.save(model.state_dict(), save_path+"part3_binary"+str(i)+str(j)+".pt")
