import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from configuration import Configuration
from voraus_ad import ANOMALY_CATEGORIES, Signals, load_torch_dataloaders

# Constants and Configurations
DETERMINISTIC_CUDA = False
DATASET_PATH = Path.home() / "Downloads" / "voraus-ad-dataset-100hz.parquet"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training configuration and hyperparameters of the model.
configuration = Configuration(
    columns="machine",
    epochs=100,
    frequencyDivider=1,
    trainGain=1.0,
    seed=177,
    batchsize=8,#32
    nCouplingBlocks=4,
    clamp=1.2,
    learningRate=5.2e-4,
    normalize=True,
    pad=True,
    nHiddenLayers=0,
    scale=2,
    kernelSize1=13,
    dilation1=2,
    kernelSize2=1,
    dilation2=1,
    kernelSize3=1,
    dilation3=1,
    milestones=[11, 61],
    gamma=0.1,
)

# Make the training reproducible.
torch.manual_seed(configuration.seed)
torch.cuda.manual_seed_all(configuration.seed)
np.random.seed(configuration.seed)
random.seed(configuration.seed)

def extract_features(data: torch.Tensor) -> np.ndarray:
    """Extract features from the dataset to use with SVM.
    
    Args:
        data (torch.Tensor): The input tensor from which to extract features.

    Returns:
        np.ndarray: Extracted features.
    """
    # Convert input tensor to numpy
    data_np = data.cpu().numpy()
    
    # Example features: Mean and Standard Deviation along the time dimension
    means = np.mean(data_np, axis=1)
    stds = np.std(data_np, axis=1)
    
    # Concatenate features (you can add more features if needed)
    features = np.concatenate([means, stds], axis=1)
    
    return features

def train_svm() -> List[Dict]:
    """Train and evaluate SVM for anomaly detection.

    Returns:
        List[Dict]: Results containing AUROC and loss per epoch.
    """
    # Load the dataset as torch data loaders.
    train_dataset, _, train_dl, test_dl = load_torch_dataloaders(
        dataset=DATASET_PATH,
        batch_size=configuration.batchsize,
        columns=Signals.groups()[configuration.columns],
        seed=configuration.seed,
        frequency_divider=configuration.frequency_divider,
        train_gain=configuration.train_gain,
        normalize=configuration.normalize,
        pad=configuration.pad,
    )

    # Collect all training data to fit the SVM
    all_features = []
    for data in train_dl:
        inputs, _ = data
        inputs = inputs.float().to(DEVICE)
        features = extract_features(inputs)
        all_features.append(features)

    # Stack all features into a single array
    X_train = np.vstack(all_features)

    # Normalize features for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Initialize and fit the One-Class SVM
    svm_model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    svm_model.fit(X_train_scaled)

    training_results: List[Dict] = []

    # Test the SVM on the test data
    for epoch in range(configuration.epochs):
        # model.eval()
        result_list = []
        for _, (tensors, labels) in enumerate(test_dl):
            inputs = tensors.float().to(DEVICE)
            features = extract_features(inputs)
            features_scaled = scaler.transform(features)
            
            # Get anomaly scores from SVM (distance to decision boundary)
            scores = -svm_model.decision_function(features_scaled)
            
            for j in range(len(scores)):
                result_labels = {k: v[j].item() if isinstance(v, torch.Tensor) else v[j] for k, v in labels.items()}
                result_labels.update(score=scores[j])
                result_list.append(result_labels)

        results = pd.DataFrame(result_list)

        # Calculate AUROC per anomaly category.
        aurocs = []
        for category in ANOMALY_CATEGORIES:
            dfn = results[(results["category"] == category.name) | (~results["anomaly"])]
            fpr, tpr, _ = metrics.roc_curve(dfn["anomaly"], dfn["score"].values, pos_label=True)
            auroc = metrics.auc(fpr, tpr)
            aurocs.append(auroc)
            print(f'{category.name}, auroc={auroc:5.3f},')

        # Calculate the AUROC mean over all categories.
        auroc_mean = np.mean(aurocs)
        training_results.append({"epoch": epoch, "aurocMean": auroc_mean})
        print(f"Epoch {epoch:0>3d}: auroc(mean)={auroc_mean:5.3f}")

    return training_results

if __name__ == "__main__":
    train_svm()
