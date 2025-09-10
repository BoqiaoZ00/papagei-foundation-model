import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch import nn
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from models.resnet import ResNet1DMoE
from linearprobing.utils import load_model_without_module_prefix

# ======================
# Local helper functions
# ======================

def load_compressed_model(pkl_path, model_arch, device="cpu"):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    compressed_layers = data.get("compressed_layers", {})
    uncompressed_params = data.get("uncompressed_params", {})

    state_dict = {}
    for layer_name, layer_data in compressed_layers.items():
        indices = layer_data["indices"]
        codebook = layer_data["codebook"]
        shape = layer_data["shape"]
        weights = codebook[indices].reshape(shape)
        state_dict[layer_name] = torch.tensor(weights, dtype=torch.float32)

    for param_name, param_value in uncompressed_params.items():
        state_dict[param_name] = torch.tensor(param_value, dtype=torch.float32)

    model = ResNet1DMoE(1, **model_arch)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model

def extract_embeddings(model, signal_tensor, device="cpu"):
    model.eval()
    with torch.no_grad():
        outputs = model(signal_tensor.to(device))
        embeddings = outputs[0].cpu().numpy()
    return embeddings

def linear_probe_regression(embeddings, labels):
    reg = LinearRegression()
    reg.fit(embeddings, labels)
    preds = reg.predict(embeddings)
    mse = mean_squared_error(labels, preds)
    r2 = r2_score(labels, preds)
    return {"mse": mse, "r2": r2}

def knn_probe_regression(embeddings, labels, k=5):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(embeddings, labels)
    preds = knn.predict(embeddings)
    mse = mean_squared_error(labels, preds)
    r2 = r2_score(labels, preds)
    return {"mse": mse, "r2": r2}

# ======================
# Main
# ======================

print("🔬 Regression demo starts")

# Load dataset
project_root = Path(__file__).parent.parent
csv_path = project_root / "data" / "kaggle" / "MI_PPG.csv"
data_csv = pd.read_csv(csv_path).values

# Separate signals and labels
signals = data_csv[:300, :-1].astype(np.float32)   # all but last column

# Manual mapping for numpy array
labels_numpy = data_csv[:300, -1]
labels = np.where(labels_numpy == "Normal", 0.0, 1.0).astype(np.float32)

print("Signals shape:", signals.shape)  # (num_segments, segment_length)
print("Labels shape:", labels.shape)


signal_tensor = torch.tensor(signals).unsqueeze(1)  # (N, 1, L)

model_config = {
    'base_filters': 32, 'kernel_size': 3, 'stride': 2, 'groups': 1,
    'n_block': 18, 'n_classes': 512, 'n_experts': 3
}
device = torch.device("cpu")

model_orig = ResNet1DMoE(1, **model_config)
model_orig = load_model_without_module_prefix(model_orig, "../weights/papagei_s.pt")
model_orig.to(device).eval()

model_int8 = load_compressed_model("../weights/papagei_int8.pkl", model_config, device)
model_cluster32 = load_compressed_model("../weights/papagei_cluster32.pkl", model_config, device)
model_int8_cluster32 = load_compressed_model("../weights/papagei_int8_cluster32.pkl", model_config, device)

emb_orig = extract_embeddings(model_orig, signal_tensor, device)
emb_int8 = extract_embeddings(model_int8, signal_tensor, device)
emb_cluster32 = extract_embeddings(model_cluster32, signal_tensor, device)
emb_int8_cluster32 = extract_embeddings(model_int8_cluster32, signal_tensor, device)

results = {}
for name, emb in {
    "Original": emb_orig,
    "INT8": emb_int8,
    "Cluster32": emb_cluster32,
    "INT8+Cluster32": emb_int8_cluster32
}.items():
    lp = linear_probe_regression(emb, labels)
    knn = knn_probe_regression(emb, labels)
    results[name] = {"linear_probe": lp, "knn": knn}

print("\n📊 Regression results")
print(f"{'Model':<20} {'LinReg MSE':<12} {'LinReg R2':<12} {'KNN MSE':<12} {'KNN R2':<12}")
print("-" * 65)
for method, res in results.items():
    print(f"{method:<20} {res['linear_probe']['mse']:<12.4f} {res['linear_probe']['r2']:<12.4f} "
          f"{res['knn']['mse']:<12.4f} {res['knn']['r2']:<12.4f}")
