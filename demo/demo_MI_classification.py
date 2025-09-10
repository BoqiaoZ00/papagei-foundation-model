import math
import os
import pickle
import time
import psutil
import copy
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.cluster import KMeans
from torch.ao.quantization import quantize_dynamic

from demo.demo_helper import evaluate_embedding_consistency, evaluate_signal_reconstruction, \
    test_reconstruction_decoder, is_signal_flat, linear_probe, evaluate_knn
from linearprobing.utils import resample_batch_signal, load_model_without_module_prefix
from preprocessing.ppg import preprocess_one_ppg_signal
from segmentations import waveform_to_segments
from models.resnet import ResNet1DMoE


# ======================
# Utility functions
# ======================

def cosine_similarity(a, b):
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    return torch.nn.functional.cosine_similarity(a, b, dim=1).mean().item()


def save_as_pkl(model, filepath):
    state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)}
    with open(filepath, 'wb') as f:
        pickle.dump(state_dict_cpu, f)


def get_current_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def estimate_param_memory_mb(model):
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters())
    return param_mem / (1024 * 1024)


# ======================
# Compressed Model class
# ======================

class CompressedModel:
    """存储聚类压缩的索引和码本"""

    def __init__(self):
        self.compressed_layers = {}
        self.uncompressed_params = {}

    def add_compressed_layer(self, layer_name, indices, codebook, original_shape):
        self.compressed_layers[layer_name] = {
            'indices': indices.astype(np.uint8),
            'codebook': codebook.astype(np.float32),
            'shape': original_shape,
        }

    def add_uncompressed_param(self, param_name, param_value):
        self.uncompressed_params[param_name] = param_value.cpu().numpy()

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'compressed_layers': self.compressed_layers,
                'uncompressed_params': self.uncompressed_params
            }, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.compressed_layers = data['compressed_layers']
        self.uncompressed_params = data['uncompressed_params']
        return self

    def decompress_to_state_dict(self):
        state_dict = {}
        for layer_name, layer_data in self.compressed_layers.items():
            indices = layer_data['indices']
            codebook = layer_data['codebook']
            shape = layer_data['shape']
            weights = codebook[indices].reshape(shape)
            state_dict[layer_name] = torch.tensor(weights, dtype=torch.float32)
        for param_name, param_value in self.uncompressed_params.items():
            state_dict[param_name] = torch.tensor(param_value, dtype=torch.float32)
        return state_dict


def estimate_compressed_memory_mb(compressed_model: CompressedModel):
    total_bytes = 0
    for layer_name, layer_data in compressed_model.compressed_layers.items():
        indices = layer_data['indices']
        codebook = layer_data['codebook']
        total_bytes += indices.nbytes
        total_bytes += codebook.nbytes
    for param_name, param_value in compressed_model.uncompressed_params.items():
        total_bytes += param_value.nbytes
    return total_bytes / (1024 * 1024)


# ======================
# Compression functions
# ======================

def compress_model_with_clustering(model, n_clusters=16, layers_to_compress=(nn.Linear, nn.Conv1d)):
    compressed_model = CompressedModel()
    reconstruction_model = copy.deepcopy(model)

    for name, module in model.named_modules():
        if isinstance(module, layers_to_compress):
            W = module.weight.data.cpu().numpy()
            shape = W.shape
            W_flat = W.reshape(-1)

            if len(np.unique(W_flat)) > n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=42)
                kmeans.fit(W_flat.reshape(-1, 1))

                centers = kmeans.cluster_centers_.squeeze()
                labels = kmeans.labels_

                compressed_model.add_compressed_layer(f"{name}.weight", labels, centers, shape)
                W_reconstructed = centers[labels].reshape(shape)
                reconstruction_model.get_submodule(name).weight.data = torch.tensor(W_reconstructed, dtype=torch.float32)
            else:
                compressed_model.add_uncompressed_param(f"{name}.weight", module.weight.data)

            if module.bias is not None:
                compressed_model.add_uncompressed_param(f"{name}.bias", module.bias.data)

    for param_name, param in model.named_parameters():
        if param_name not in compressed_model.uncompressed_params and f"{param_name}" not in compressed_model.compressed_layers:
            compressed_model.add_uncompressed_param(param_name, param.data)

    return compressed_model, reconstruction_model


# ======================
# Evaluation
# ======================

def evaluate_model(model_eval, signal_tensor, emb_orig, device, num_runs=50, compressed_model=None):
    """评估模型性能、延迟和内存 (包括压缩和解压后的大小)"""
    model_eval.to(device)
    model_eval.eval()

    # Warm-up
    with torch.inference_mode():
        _ = model_eval(signal_tensor)

    # Latency
    start = time.time()
    for _ in range(num_runs):
        with torch.inference_mode():
            _ = model_eval(signal_tensor)
    avg_latency = (time.time() - start) / num_runs

    # Memory usage (compressed vs decompressed)
    if compressed_model is not None:
        mem_usage_compressed = estimate_compressed_memory_mb(compressed_model)
    else:
        mem_usage_compressed = estimate_param_memory_mb(model_eval)

    # Always measure the decompressed model param memory (true PyTorch tensors in RAM)
    mem_usage_decompressed = sum(p.numel() * p.element_size() for p in model_eval.parameters()) / (1024 ** 2)

    # Similarity
    with torch.inference_mode():
        outputs = model_eval(signal_tensor)
        emb_eval = outputs[0].cpu().detach().numpy()
    similarity = cosine_similarity(emb_orig, emb_eval)

    # Embedding consistency
    consistency_results = evaluate_embedding_consistency(emb_orig, emb_eval)

    # Signal reconstruction
    # reconstruction_results = evaluate_signal_reconstruction(
    #     emb_orig,
    #     emb_eval,
    #     signal_tensor,
    #     device=device
    # )

    linear_probe_result = linear_probe(model_eval, signal_tensor, labels, device)

    knn_result = evaluate_knn(emb_eval, labels)

    print("Embedding shape:", emb_orig.shape)
    print("Any NaN:", np.isnan(emb_orig).any())
    print("Any Inf:", np.isinf(emb_orig).any())
    print("Min:", np.min(emb_orig), "Max:", np.max(emb_orig), "Mean:", np.mean(emb_orig))

    def check_for_nans(model, x):
        nan_layers = []

        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                if torch.isnan(output).any():
                    nan_layers.append((module.__class__.__name__, output.shape))

        hooks = []
        for name, module in model.named_modules():
            hooks.append(module.register_forward_hook(hook_fn))

        with torch.no_grad():
            _ = model(x)

        for h in hooks:
            h.remove()

        return nan_layers

    # Usage
    nan_layers = check_for_nans(model, signal_tensor[:1])  # check first batch
    print("Layers with NaN outputs:", nan_layers)

    return {
        "similarity": similarity,
        "latency_ms": avg_latency * 1000,
        "mem_usage_compressed_mb": mem_usage_compressed,
        "mem_usage_decompressed_mb": mem_usage_decompressed,
        "consistency: mean_norm_shift": consistency_results["mean_norm_shift"],
        "consistency: std_norm_shift": consistency_results["std_norm_shift"],
        "consistency: wasserstein_norm": consistency_results["wasserstein_norm"],
        "consistency: pairwise_similarity_corr": consistency_results["pairwise_similarity_corr"],
        "linear_probe_acc": linear_probe_result["linear_probe_acc"],
        "linear_probe_f1": linear_probe_result["linear_probe_f1"],
        "knn_acc": knn_result["knn_acc"],
    }



# ======================
# Comparison function
# ======================

def compare_all_methods(model, signal_tensor, emb_orig, device):
    results = {}
    original_pkl_path = "../weights/papagei_s.pkl"
    save_as_pkl(model, original_pkl_path)
    original_size = os.path.getsize(original_pkl_path) / (1024 * 1024)

    # 1. Baseline
    results["原始模型"] = {
        "compression_ratio": 0.0,
        "file_size_mb": original_size,
        **evaluate_model(model, signal_tensor, emb_orig, device)
    }

    # 2. INT8
    torch.backends.quantized.engine = "qnnpack"
    model_int8 = quantize_dynamic(copy.deepcopy(model), {nn.Linear}, dtype=torch.qint8)
    int8_pkl = "../weights/papagei_int8.pkl"
    save_as_pkl(model_int8, int8_pkl)
    int8_size = os.path.getsize(int8_pkl) / (1024 * 1024)
    results["INT8量化"] = {
        "compression_ratio": (original_size - int8_size) / original_size,
        "file_size_mb": int8_size,
        **evaluate_model(model_int8, signal_tensor, emb_orig, device)
    }

    # 3. Clustering (32)
    compressed_model, reconstructed_model = compress_model_with_clustering(model, n_clusters=32)
    cluster_pkl = "../weights/papagei_cluster32.pkl"
    compressed_model.save(cluster_pkl)
    cluster_size = os.path.getsize(cluster_pkl) / (1024 * 1024)
    results["聚类32"] = {
        "compression_ratio": (original_size - cluster_size) / original_size,
        "file_size_mb": cluster_size,
        **evaluate_model(reconstructed_model, signal_tensor, emb_orig, device, compressed_model=compressed_model)
    }

    # 4. INT8 + Clustering (32)
    compressed_int8_model, reconstructed_int8_clustered = compress_model_with_clustering(model_int8, n_clusters=32)
    int8_cluster_pkl = "../weights/papagei_int8_cluster32.pkl"
    compressed_int8_model.save(int8_cluster_pkl)
    int8_cluster_size = os.path.getsize(int8_cluster_pkl) / (1024 * 1024)
    results["INT8+聚类32"] = {
        "compression_ratio": (original_size - int8_cluster_size) / original_size,
        "file_size_mb": int8_cluster_size,
        **evaluate_model(reconstructed_int8_clustered, signal_tensor, emb_orig, device, compressed_model=compressed_int8_model)
    }

    # --- Get embeddings for extra quality checks (Still under testing) ---
    with torch.inference_mode():
        emb_int8_cluster32 = reconstructed_int8_clustered(signal_tensor)[0].cpu().numpy()

    # --- 1. Embedding consistency ---
    # consistency_results = evaluate_embedding_consistency(emb_orig, emb_int8_cluster32)
    #
    # # --- 2. Signal reconstruction ---
    # reconstruction_results = evaluate_signal_reconstruction(
    #     emb_orig,
    #     emb_int8_cluster32,
    #     signal_tensor,
    #     device=device
    # )
    # print("Embedding Consistency:", consistency_results)
    # print("Signal Reconstruction:", reconstruction_results)

    # test_reconstruction_decoder(
    #     emb_orig,
    #     emb_int8_cluster32,
    #     signal_tensor,
    #     device=device
    # )

    # Summary
    print("\n📊 压缩方法综合对比总结")
    print(f"{'方法':<10} {'文件大小(MB)':<15} {'压缩比例':<10} {'内存占用(MB)-压缩':<15} {'内存占用(MB)-解压':<15} {'推理延迟(ms)':<15} {'相似度':<15}{'mean_norm_shift':<15} {'std_norm_shift':<15} {'corr':<15} {'linear_probe_acc':<15} {'linear_probe_f1':<15} {'knn_acc':<15}")
    print("-" * 75)
    for method, res in results.items():
        print(f"{method:<15} {res['file_size_mb']:<15.4f} {res['compression_ratio']:<15.2%} {res['mem_usage_compressed_mb']:<15.4f} {res['mem_usage_decompressed_mb']:<15.4f} {res['latency_ms']:<15.4f} {res['similarity']:<15.4f} {res['consistency: mean_norm_shift']:<15.4f} {res['consistency: std_norm_shift']:<15.4f} {res['consistency: pairwise_similarity_corr']:<15.4f} {res['linear_probe_acc']:<15.8f} {res['linear_probe_f1']:<15.8f} {res['knn_acc']:<15.4f}")

    return results


# ======================
# Main demo
# ======================

print("Demo starts")
model_config = {
    'base_filters': 32, 'kernel_size': 3, 'stride': 2, 'groups': 1,
    'n_block': 18, 'n_classes': 512, 'n_experts': 3
}
model = ResNet1DMoE(1, **model_config)
model = load_model_without_module_prefix(model, "../weights/papagei_s.pt")
device = torch.device("cpu")
model.to(device).eval()

# # Fake signal
# fs, fs_target, seg_len, total_len = 500, 125, 10, 600
# signal = np.random.randn(total_len * fs)
# signal_processed, _, _, _ = preprocess_one_ppg_signal(signal, fs)
# segments = waveform_to_segments('ppg', fs * seg_len, signal_processed)
# resampled = resample_batch_signal(segments, fs, fs_target, axis=-1)
# signal_tensor = torch.Tensor(resampled).unsqueeze(1).to(device)
# print("signal_tensor.shape: ", signal_tensor.shape)

# Real signal (VitalDB)
# import vitaldb
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
#
# device = torch.device("cpu")
#
# # ======================
# # Load real PPG data
# # ======================
# caseid = 102   # Replace with your desired case id
# tracks = ["SNUADC/PLETH"]
#
# # Load the waveforms at 100 Hz
# fs = 500   # sampling frequency
# fs_target = 125   # target sampling frequency for model
# seg_len = 10      # seconds per segment
#
# data = vitaldb.load_case(caseid, tracks, 1/fs)
# print("Raw data shape:", data.shape)
#
# ppg_signal = data[:, 0]
# t_axis = np.arange(len(ppg_signal)) / fs   # renamed to avoid conflict with time module
#
# # ======================
# # Preprocess + segment
# # ======================
#
# # Preprocess signal
# signal_processed, _, _, _ = preprocess_one_ppg_signal(ppg_signal, fs)
#
# # Segment into fixed-length windows
# segment_length = fs * seg_len
# segments = waveform_to_segments('ppg', segment_length, signal_processed)
#
# # Resample to target frequency
# resampled = resample_batch_signal(segments, fs, fs_target, axis=-1)
#
# clean_segments = []
# for seg in resampled:  # each segment
#     if not is_signal_flat(seg, fs_target):  # keep only good signals
#         clean_segments.append(seg)
#
# signal_tensor = torch.Tensor(clean_segments).unsqueeze(1).to(device)
# # signal_tensor = signal_tensor + 1e-6 * torch.randn_like(signal_tensor)
# print("signal_tensor.shape:", signal_tensor.shape)
#
# # ======================
# # Optional: visualize a segment
# # ======================
# print("PPG shape:", ppg_signal.shape, "Duration (s):", len(ppg_signal)/fs)
#
# # Plot first 10 seconds
# plt.figure(figsize=(12, 4))
# plt.plot(t_axis[:fs*10], ppg_signal[:fs*10], 'b')
# plt.title("PPG Signal (first 10 seconds)")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Real Signal (Kaggle)
# Load dataset
project_root = Path(__file__).parent.parent
csv_path = project_root / "data" / "kaggle" / "MI_PPG.csv"
data_csv = pd.read_csv(csv_path).values

# Separate signals and labels
signals = data_csv[:500, :-1].astype(np.float32)   # all but last column
labels = data_csv[:500, -1]                        # last column is label

print("Signals shape:", signals.shape)  # (num_segments, segment_length)
print("Labels shape:", labels.shape)

# Convert to tensor (PyTorch expects [batch, channel, length])
signal_tensor = torch.tensor(signals).unsqueeze(1)  # add channel dim

print("signal_tensor.shape:", signal_tensor.shape)  # e.g. (N, 1, 2000)

with torch.inference_mode():
    emb_orig = model(signal_tensor)[0].cpu().detach().numpy()

# Run comparisons
all_results = compare_all_methods(model, signal_tensor, emb_orig, device)
