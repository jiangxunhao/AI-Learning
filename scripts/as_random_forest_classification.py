from sklearn.ensemble import RandomForestClassifier
import numpy as np
import argparse

from util import preparation


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained AS model on a test set")
    parser.add_argument("--train", type=str, required=True, help="Path to a train dataset")
    parser.add_argument("--test", type=str, required=True, help="Path to a test dataset")

    args = parser.parse_args()

    print(f"\nConstructing a random forests for {args.train} and evaluating it on {args.test}")

    # load the train set and preprocess the data
    features_tr, performances_train = preparation.load(args.train)
    features_train = preparation.standardise(features_tr)
    labels_train = np.argmin(performances_train, axis=1)

    # construct a random forest from the train set
    classifier = RandomForestClassifier()
    classifier.fit(features_train, labels_train)

    # load the test set
    features_te, performances_test = preparation.load(args.test)
    mean = np.loadtxt("mean.txt")
    std = np.loadtxt("std.txt")
    features_test = preparation.standardise(features_te, mean, std)
    labels_test = np.argmin(performances_test, axis=1)

    pred_labels = classifier.predict(features_test)

    n = features_test.shape[0]
    n_corrects = (pred_labels == labels_test).sum()

    accuracy = n_corrects / n
    avg_cost = np.mean(performances_test[np.arange(len(pred_labels)), pred_labels])
    sbs_avg_cost = np.min(np.mean(performances_test, axis=0))
    vbs_avg_cost = np.mean(np.min(performances_test, axis=1))
    sbs_vbs_gap = (avg_cost - vbs_avg_cost) / (sbs_avg_cost - vbs_avg_cost)
    print(f"\nFinal results: accuracy: {accuracy:4.4f}, \tavg_cost: {avg_cost:8.4f}, \tsbs_cost: {sbs_avg_cost:8.4f}, \tvbs_cost: {vbs_avg_cost:8.4f}, \tsbs_vbs_gap: {sbs_vbs_gap:2.4f}")

    print(f"\nModel finished")

if __name__ == "__main__":
    main()
