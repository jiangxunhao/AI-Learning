import argparse
import numpy as np
import torch

from util import preparation
from asmodel import as_nn_regression, as_nn_classification_basic, as_nn_classification_advanced, as_nn_binary_classification


def main():
    parser = argparse.ArgumentParser(description="Train an AS model and save it to file")
    parser.add_argument("--model-type", type=str, required=True, help="Path to a trained AS model (a .pt file)")
    parser.add_argument("--data", type=str, required=True, help="Path to a dataset")
    parser.add_argument("--save", type=str, required=True,
                        help="Save the trained model (and any related info) to a .pt file")

    args = parser.parse_args()

    print(f"\nTraining a {args.model_type} model on {args.data}, and save it to {args.save}")

    features, performances = preparation.load(args.data)
    save_path = args.save
    """
    plt.plot(features[..., 1])
    plt.show()
    """

    # get the indices of constant columns in features
    # constant_cols = preparation.constant_columns(features)

    # remove the constant columns of features
    # non_constant_features = preparation.remove_constant(features, constant_cols)

    # standardise the feature data and performance data
    standard_features = preparation.standardise(features)
    standard_performances = performances

    # transform the type of data to follow the requirement of torch
    features_train = torch.tensor(standard_features.astype(np.float32))
    performances_train = torch.tensor(standard_performances.astype(np.float32))

    print(features_train.shape)
    print(performances_train.shape)

    if args.model_type == "regresion_nn":
        as_nn_regression.as_regression_train(features_train, performances_train, save_path)

    elif args.model_type == "classification_nn":
        as_nn_classification_basic.as_classification_basic_train(features_train, performances_train, save_path)

    elif args.model_type == "classification_nn_cost":
        as_nn_classification_advanced.as_classification_advanced_train(features_train, performances_train, save_path)

    elif args.model_type == "binary_classification_nn":
        as_nn_binary_classification.as_binary_classification_train(features_train, performances_train, save_path)

    # print results
    print(f"\nTraining finished")


if __name__ == "__main__":
    main()
