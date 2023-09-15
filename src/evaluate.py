import argparse
import numpy as np
import torch
import torch.nn as nn

from util import preparation, P4util
from asmodel import as_nn_regression, as_nn_classification_basic, as_nn_classification_advanced, as_nn_binary_classification


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained AS model on a test set")
    parser.add_argument("--model", type=str, required=True, help="Path to a trained AS model (a .pt file)")
    parser.add_argument("--data", type=str, required=True, help="Path to a dataset")
    
    args = parser.parse_args()

    print(f"\nLoading trained model {args.model} and evaluating it on {args.data}")

    load_path = args.model

    features, performances = preparation.load(args.data)
    mean = np.loadtxt("mean.txt")
    std = np.loadtxt("std.txt")

    standard_features = preparation.standardise(features, mean, std)

    features_test = torch.tensor(standard_features.astype(np.float32))
    performances_test = torch.tensor(performances.astype(np.float32))
    true_chosen = torch.argmin(performances_test, dim=1)
    n = features_test.shape[0]

    # load the given model, make predictions on the given dataset and evaluate the model's performance.
    # Your evaluation should report four evaluation metrics: avg_loss, accuracy, avg_cost, sbs_vbs_gap (as listed below)
    # you should also calculate the average cost of the SBS and the VBS
    avg_loss = np.inf # the average loss value across the given dataset
    accuracy = 0 # classification accuracy 
    avg_cost = np.inf # the average cost of the predicted algorithms on the given dataset
    sbs_vbs_gap = np.inf # the SBS-VBS gap of your model on the given dataset
    sbs_avg_cost = np.inf # the average cost of the SBS on the given dataset 
    vbs_avg_cost = np.inf # the average cost of the VBS on the given dataset
    # YOUR CODE HERE

    if args.model == "models/part1.pt":
        model = as_nn_regression.ASRegression(features_test.shape[1], performances_test.shape[1])
        model.load_state_dict(torch.load(load_path))
        model.eval()

        pred_performances = model(features_test)
        pred_chosen = torch.argmin(pred_performances, dim=1)

        loss = nn.functional.mse_loss(pred_performances, performances_test, reduction="mean")
        n_corrects = (pred_chosen == true_chosen).sum()

        avg_loss = loss / n
        accuracy = n_corrects / n
        avg_cost = P4util.avg_cost(pred_chosen, performances_test)
        sbs_vbs_gap = P4util.sbs_vbs_gap(pred_chosen, performances_test)
        sbs_avg_cost = P4util.sbs_avg_cost(performances_test)
        vbs_avg_cost = P4util.vbs_avg_cost(performances_test)

    elif args.model == "models/part2_basic.pt":
        model = as_nn_classification_basic.ASClassificationBasic(features_test.shape[1], performances_test.shape[1])
        model.load_state_dict(torch.load(load_path))
        model.eval()

        pred = model(features_test)
        pred_chosen = pred.argmax(dim=1)

        loss = nn.functional.cross_entropy(pred, true_chosen, reduction="mean")
        n_corrects = (pred_chosen == true_chosen).sum()

        avg_loss = loss / n
        accuracy = n_corrects / n
        avg_cost = P4util.avg_cost(pred_chosen, performances_test)
        sbs_vbs_gap = P4util.sbs_vbs_gap(pred_chosen, performances_test)
        sbs_avg_cost = P4util.sbs_avg_cost(performances_test)
        vbs_avg_cost = P4util.vbs_avg_cost(performances_test)

    elif args.model == "models/part2_advanced.pt":
        model = as_nn_classification_advanced.ASClassificationAdvanced(features_test.shape[1], performances_test.shape[1])
        model.load_state_dict(torch.load(load_path))
        model.eval()

        pred = model(features_test)
        pred_chosen = pred.argmax(dim=1)

        loss = as_nn_classification_advanced.RegretLoss().forward(pred, true_chosen, performances_test)
        n_corrects = (pred_chosen == true_chosen).sum()

        avg_loss = loss
        accuracy = n_corrects / n
        avg_cost = P4util.avg_cost(pred_chosen, performances_test)
        sbs_vbs_gap = P4util.sbs_vbs_gap(pred_chosen, performances_test)
        sbs_avg_cost = P4util.sbs_avg_cost(performances_test)
        vbs_avg_cost = P4util.vbs_avg_cost(performances_test)

    elif args.model == "models/Part3/":
        pred_vote = torch.zeros(performances_test.shape)
        loss = 0
        for i in range(performances_test.shape[1]):
            for j in range(i+1, performances_test.shape[1]):
                model = as_nn_binary_classification.ASBinaryClassification(features_test.shape[1], 2)
                model.load_state_dict(torch.load(load_path+"part3_binary"+str(i)+str(j)+".pt"))
                model.eval()

                pred = model(features_test)
                labels_test = torch.argmin(performances_test[:, (i, j)], dim=1)
                binary_performances_test = performances_test[:, (i, j)]

                loss_function = as_nn_binary_classification.RegretLoss()
                loss += loss_function(pred, labels_test, binary_performances_test)

                for index in range(pred_vote.shape[0]):
                    if pred[index, 0] > pred[index, 1]:
                        pred_vote[index, i] += 1
                    else:
                        pred_vote[index, j] += 1

        pred_chosen = torch.argmax(pred_vote, dim=1)
        n_corrects = (pred_chosen == true_chosen).sum()

        avg_loss = loss / n
        accuracy = n_corrects / n
        avg_cost = P4util.avg_cost(pred_chosen, performances_test)
        sbs_vbs_gap = P4util.sbs_vbs_gap(pred_chosen, performances_test)
        sbs_avg_cost = P4util.sbs_avg_cost(performances_test)
        vbs_avg_cost = P4util.vbs_avg_cost(performances_test)

    # print results
    print(f"\nFinal results: loss: {avg_loss:8.4f}, \taccuracy: {accuracy:4.4f}, \tavg_cost: {avg_cost:8.4f}, \tsbs_cost: {sbs_avg_cost:8.4f}, \tvbs_cost: {vbs_avg_cost:8.4f}, \tsbs_vbs_gap: {sbs_vbs_gap:2.4f}")


if __name__ == "__main__":
    main()
