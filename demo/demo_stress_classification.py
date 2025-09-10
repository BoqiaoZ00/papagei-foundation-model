import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch import nn
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from models.resnet import ResNet1DMoE
from linearprobing.utils import load_model_without_module_prefix


# ======================
# Local helper functions
# ======================

def load_compressed_model(pkl_path, model_arch, device="cpu"):
    """Load a compressed model from pickle file and return a ResNet instance with decompressed weights."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    compressed_layers = data.get("compressed_layers", {})
    uncompressed_params = data.get("uncompressed_params", {})

    # Build a state dict
    state_dict = {}
    for layer_name, layer_data in compressed_layers.items():
        indices = layer_data["indices"]
        codebook = layer_data["codebook"]
        shape = layer_data["shape"]
        weights = codebook[indices].reshape(shape)
        state_dict[layer_name] = torch.tensor(weights, dtype=torch.float32)

    for param_name, param_value in uncompressed_params.items():
        state_dict[param_name] = torch.tensor(param_value, dtype=torch.float32)

    # Create model and load weights
    model = ResNet1DMoE(1, **model_arch)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model


def extract_embeddings(model, signal_tensor, device="cpu"):
    """Extract embeddings from the model given PPG signals."""
    model.eval()
    with torch.no_grad():
        outputs = model(signal_tensor.to(device))
        embeddings = outputs[0].cpu().numpy()
    return embeddings


def linear_probe(embeddings, labels):
    """Run a simple linear classifier (logistic regression)"""
    clf = LogisticRegression(max_iter=50, solver="liblinear")
    clf.fit(embeddings, labels)
    preds = clf.predict(embeddings)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"acc": acc, "f1": f1}


def knn_probe(embeddings, labels, k=3):
    """Run KNN classification"""
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(embeddings, labels)
    preds = knn.predict(embeddings)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"acc": acc, "f1": f1}


# ======================
# Main
# ======================

print("New classification demo starts")

# Dataset
project_root = Path(__file__).parent.parent
csv_path = project_root / "data" / "kaggle" / "mental_stress_ppg.csv"
data_csv = pd.read_csv(csv_path)

# Use subset (55 samples, starting from col=2 onward as signal, col=1 as label)
# Drop the first two columns if they are IDs/labels
signals_raw = data_csv.iloc[:55, 2:]   # dataframe, keep as df
# Find min valid length across rows (non-NaN count)
min_len = signals_raw.notna().sum(axis=1).min()
print("Min valid length across signals:", min_len)
# Truncate all rows to this length
signals = signals_raw.iloc[:, :min_len].to_numpy(dtype=np.float32)
labels = data_csv.iloc[:55, 1].values  # assuming column 1 is label
print("Final signals shape:", signals.shape)  # (55, min_len)

# Convert to tensor for model
signal_tensor = torch.tensor(signals).unsqueeze(1)  # (N, 1, L)

# Model config (must match training)
model_config = {
    'base_filters': 32, 'kernel_size': 3, 'stride': 2, 'groups': 1,
    'n_block': 18, 'n_classes': 512, 'n_experts': 3
}
device = torch.device("cpu")

# --- Load original model ---
model_orig = ResNet1DMoE(1, **model_config)
model_orig = load_model_without_module_prefix(model_orig, "../weights/papagei_s.pt")
model_orig.to(device).eval()

# --- Load compressed models from disk ---
model_int8 = load_compressed_model("../weights/papagei_int8.pkl", model_config, device)
model_cluster32 = load_compressed_model("../weights/papagei_cluster32.pkl", model_config, device)
model_int8_cluster32 = load_compressed_model("../weights/papagei_int8_cluster32.pkl", model_config, device)

# --- Extract embeddings ---
emb_orig = extract_embeddings(model_orig, signal_tensor, device)
emb_int8 = extract_embeddings(model_int8, signal_tensor, device)
emb_cluster32 = extract_embeddings(model_cluster32, signal_tensor, device)
emb_int8_cluster32 = extract_embeddings(model_int8_cluster32, signal_tensor, device)

# --- Evaluate classification ---
results = {}
for name, emb in {
    "Original": emb_orig,
    "INT8": emb_int8,
    "Cluster32": emb_cluster32,
    "INT8+Cluster32": emb_int8_cluster32
}.items():
    lp = linear_probe(emb, labels)
    knn = knn_probe(emb, labels)
    results[name] = {"linear_probe": lp, "knn": knn}

# --- Print summary ---
print("\n📊 Classification results")
print(f"{'Model':<20} {'LogReg Acc':<12} {'LogReg F1':<12} {'KNN Acc':<12} {'KNN F1':<12}")
print("-" * 65)
for method, res in results.items():
    print(f"{method:<20} {res['linear_probe']['acc']:<12.4f} {res['linear_probe']['f1']:<12.4f} "
          f"{res['knn']['acc']:<12.4f} {res['knn']['f1']:<12.4f}")
