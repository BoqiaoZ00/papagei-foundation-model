import math
import os
import pickle

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.ao.quantization import quantize_dynamic
from torch.nn.utils import prune

from linearprobing.utils import resample_batch_signal, load_model_without_module_prefix
from preprocessing.ppg import preprocess_one_ppg_signal
from segmentations import waveform_to_segments
from torch_ecg._preprocessors import Normalize
from models.resnet import ResNet1DMoE
import copy


def cosine_similarity(a, b):
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    sim = torch.nn.functional.cosine_similarity(a, b, dim=1)
    return sim.mean().item()


def calculate_model_size(model):
    """计算模型参数量和实际占用的内存"""
    total_params = 0
    total_size_bytes = 0
    zero_params = 0

    for param in model.parameters():
        total_params += param.numel()
        total_size_bytes += param.numel() * param.element_size()
        zero_params += (param == 0).sum().item()

    return total_params, total_size_bytes, zero_params


print("Demo starts")
print("=" * 50)

# Define Model Configuration
model_config = {
    'base_filters': 32,
    'kernel_size': 3,
    'stride': 2,
    'groups': 1,
    'n_block': 18,
    'n_classes': 512,  # Embedding dimension
    'n_experts': 3
}

# Initialize Model
model = ResNet1DMoE(
    in_channels=1,
    base_filters=model_config['base_filters'],
    kernel_size=model_config['kernel_size'],
    stride=model_config['stride'],
    groups=model_config['groups'],
    n_block=model_config['n_block'],
    n_classes=model_config['n_classes'],
    n_experts=model_config['n_experts']
)

# Load Pre-trained Weights
model_path = "../weights/papagei_s.pt"  # Ensure this path is correct
model = load_model_without_module_prefix(model, model_path)
device = "cpu"

model.to(device)

model.eval()  # Set model to evaluation mode

print(f"Model loaded on {device}")
print("=" * 50)

# Random Example PPG Signal
fs = 500  # Original sampling frequency in Hz
fs_target = 125  # Target sampling frequency in Hz
segment_duration_seconds = 10  # Duration of each segment in seconds
signal_duration_seconds = 60  # Total duration of the example signal

signal = np.random.randn(signal_duration_seconds * fs)  # Example: 60s signal at 500Hz
print(f"Original PPG dimensions: {signal.shape}")

# Clean and segment the signal
signal_processed, _, _, _ = preprocess_one_ppg_signal(waveform=signal, frequency=fs)

segment_length_original_fs = fs * segment_duration_seconds
segmented_signals = waveform_to_segments(
    waveform_name='ppg',  # Can be any name, not strictly used in this function
    segment_length=segment_length_original_fs,
    clean_signal=signal_processed
)

# Resample segments
resampled_segments = resample_batch_signal(
    segmented_signals,
    fs_original=fs,
    fs_target=fs_target,
    axis=-1
)
print(f"After segmentation and resampling: {resampled_segments.shape}")  # (num_segments, segment_length_target_fs)

# Convert to PyTorch Tensor
signal_tensor = torch.Tensor(resampled_segments).unsqueeze(dim=1).to(
    device)  # (num_segments, 1, segment_length_target_fs)

with torch.inference_mode():
    outputs = model(signal_tensor)
    # PaPaGei-S returns a tuple (embeddings, expert_outputs, gating_weights)
    # We are interested in the first element: embeddings
    emb_orig = outputs[0].cpu().detach().numpy()
print(f"Embedding dimensions: {emb_orig.shape}")  # (num_segments, n_classes)

print("=" * 50)
print("Compression starts")
print("=" * 50)

# ################## quantize
# print("Compress to int8")
# print("=" * 50)
# # 设置 quantization backend
# torch.backends.quantized.engine = "qnnpack"  # CPU 上推荐 qnnpack，x86 可用 fbgemm
#
# # 只量化 Linear
# quantized_model = quantize_dynamic(
#     copy.deepcopy(model),
#     {torch.nn.Linear},  # 只能放 Linear
#     dtype=torch.qint8
# )
#
# torch.save(quantized_model.state_dict(), "../weights/papagei_int8.pt")
#
# # 比较文件大小
# size_original = os.path.getsize("../weights/papagei_s.pt") / (1024*1024)  # MB
# size_int8 = os.path.getsize("../weights/papagei_int8.pt") / (1024*1024)  # MB
# print(f"压缩率: {(size_original-size_int8)/size_original * 100:.2f}%")
# # 比较acc
# with torch.inference_mode():
#     outputs_pruned = quantized_model(signal_tensor)
#     emb_quantized = outputs_pruned[0].cpu().detach().numpy()
# sim = cosine_similarity(emb_orig, emb_quantized)
# print(f"original_quantized_平均余弦相似度: {sim:.4f}")


# ################## 结构性剪枝
# print("=" * 50)
# print("Structure prune starts")
# def structured_prune_model(model, amount=0.1):
#     """
#     对 Conv1d 层做结构化剪枝，剪掉 amount 比例的输出通道
#     """
#     for name, module in model.named_modules():
#         if isinstance(module, torch.nn.Conv1d):
#             # 对 weight 做 L1 范数排序，剪掉 dim=0 (输出通道)
#             prune.ln_structured(module, name="weight", amount=amount, n=1, dim=0)
#             prune.remove(module, "weight")  # 移除掩码，真正变小
#     return model
#
# pruned_model = structured_prune_model(model, amount=0.1)  # 剪 10% 通道
#
# torch.save(pruned_model.state_dict(), "../weights/papagei_struct_pruned.pt")
# size_original = os.path.getsize("../weights/papagei_s.pt") / (1024*1024)  # MB
# size_pruned = os.path.getsize("../weights/papagei_struct_pruned.pt") / (1024*1024)  # MB
# print(f"压缩率: {(size_original-size_pruned)/size_original * 100:.2f}%")
# with torch.inference_mode():
#     outputs_pruned = pruned_model(signal_tensor)
#     emb_pruned = outputs_pruned[0].cpu().detach().numpy()
# sim = cosine_similarity(emb_orig, emb_pruned)
# print(f"original_structure_pruned_平均余弦相似度: {sim:.4f}")
#
#
#
# ################## 非结构化剪枝
# print("=" * 50)
# print("Un-structure prune starts")
#
# def unstructured_prune_model(model, amount=0.2):
#     """
#     对模型做非结构化剪枝，剪掉 amount 比例的权重
#     """
#     parameters_to_prune = []
#
#     for name, module in model.named_modules():
#         if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
#             parameters_to_prune.append((module, 'weight'))
#             # 如果有bias，也可以剪枝
#             if hasattr(module, 'bias') and module.bias is not None:
#                 parameters_to_prune.append((module, 'bias'))
#
#     # 全局剪枝 - 在所有层中选择最小的权重进行剪枝
#     prune.global_unstructured(
#         parameters_to_prune,
#         pruning_method=prune.L1Unstructured,
#         amount=amount,
#     )
#
#     # 移除剪枝掩码，使权重真正变为0
#     for module, param_name in parameters_to_prune:
#         prune.remove(module, param_name)
#
#     return model
#
# import copy
# model_unstructure_pruned = copy.deepcopy(model)
# model_unstructure_pruned = unstructured_prune_model(model_unstructure_pruned, amount=0.01)
# torch.save(model_unstructure_pruned.state_dict(), "../weights/papagei_unstruct_pruned.pt")
#
# size_original = os.path.getsize("../weights/papagei_s.pt") / (1024*1024)  # MB
# size_unstruct_pruned = os.path.getsize("../weights/papagei_unstruct_pruned.pt") / (1024*1024)  # MB
# print(f"压缩率: {(size_original-size_unstruct_pruned)/size_original * 100:.2f}%")
#
# unstruct_pruned_params, pruned_size_bytes, unstruct_prune_zero_params = calculate_model_size(model_unstructure_pruned)
# actual_sparsity = unstruct_prune_zero_params / unstruct_pruned_params
# print(f"非结构性剪枝后稀疏度: {actual_sparsity:.1%}")
# print(f"零参数数量: {unstruct_prune_zero_params:,}")
# orig_params, orig_size_bytes, orig_zero_params = calculate_model_size(model)
# print(f"original稀疏度: {orig_zero_params/orig_params:.1%}")
#
# with torch.inference_mode():
#     outputs_pruned = model_unstructure_pruned(signal_tensor)
#     emb_pruned = outputs_pruned[0].cpu().detach().numpy()
# sim = cosine_similarity(emb_orig, emb_pruned)
# print(f"original_unstructure_pruned_平均余弦相似度: {sim:.4f}")
#
#
# ############# low rank decomposition
# def decompose_linear_layer(linear: nn.Linear, rank: int):
#     """
#     把 linear (out_features, in_features) 分解为:
#       first:  nn.Linear(in_features, rank, bias=False)
#       second: nn.Linear(rank, out_features, bias=True)  (second 保留原 bias)
#     返回 nn.Sequential(first, second)
#     """
#     W = linear.weight.data.clone()  # shape (out, in)
#     # SVD
#     U, S, Vh = torch.linalg.svd(W, full_matrices=False)  # U:(out, r0), Vh:(r0, in)
#     # 截取 top-rank
#     U_r = U[:, :rank]        # (out, rank)
#     S_r = S[:rank]           # (rank,)
#     Vh_r = Vh[:rank, :]      # (rank, in)
#
#     first = nn.Linear(linear.in_features, rank, bias=False)
#     second = nn.Linear(rank, linear.out_features, bias=True)
#
#     # first.weight shape: (rank, in)
#     first.weight.data = Vh_r.clone()
#     # second.weight shape: (out, rank)
#     second.weight.data = (U_r * S_r.unsqueeze(0)).clone()
#     # bias: move original bias to second
#     if linear.bias is not None:
#         second.bias.data = linear.bias.data.clone()
#     else:
#         second.bias.data.zero_()
#
#     return nn.Sequential(first, second)
#
# def decompose_conv1d_pointwise(conv: nn.Conv1d, rank: int):
#     """
#     仅对 kernel_size == 1 的 pointwise conv 做分解：
#     Conv1d(in, out, k=1) -> Conv1d(in, rank, k=1) -> Conv1d(rank, out, k=1)
#     保留 bias 到最后一层。
#     """
#     assert conv.kernel_size == (1,) or conv.kernel_size == 1, "只处理 pointwise conv (kernel=1)"
#     W = conv.weight.data.clone()  # shape (out, in, 1)
#     out_ch, in_ch = W.shape[0], W.shape[1]
#     W2d = W.view(out_ch, in_ch)   # (out, in)
#     # SVD
#     U, S, Vh = torch.linalg.svd(W2d, full_matrices=False)
#     U_r = U[:, :rank]
#     S_r = S[:rank]
#     Vh_r = Vh[:rank, :]
#
#     conv1 = nn.Conv1d(in_channels=in_ch, out_channels=rank, kernel_size=1, bias=False)
#     conv2 = nn.Conv1d(in_channels=rank, out_channels=out_ch, kernel_size=1, bias=(conv.bias is not None))
#
#     conv1.weight.data = Vh_r.clone().view(rank, in_ch, 1)
#     conv2.weight.data = (U_r * S_r.unsqueeze(0)).clone().view(out_ch, rank, 1)
#     if conv.bias is not None:
#         conv2.bias.data = conv.bias.data.clone()
#     return nn.Sequential(conv1, conv2)
#
# def low_rank_decompose_model(model: nn.Module, rank_ratio: float = 0.25, linear_min_size: int = 512):
#     """
#     遍历 model 的子模块，对 Linear 层和 kernel=1 Conv1d 层做低秩分解。
#     - rank_ratio: 保留秩占 min(in,out) 的比例，例如 0.25
#     - linear_min_size: 只分解大于该阈值的 Linear（避免分解小矩阵导致反而更慢）
#     返回新的 model（深拷贝替换）
#     """
#     model_new = copy.deepcopy(model)
#     for name, module in list(model_new.named_modules()):
#         if name == "":  # skip root
#             continue
#         path = name.split('.')
#         parent = model_new
#         for p in path[:-1]:
#             parent = getattr(parent, p)
#         last_name = path[-1]
#
#         mod = getattr(parent, last_name)
#
#         # Use fixed ratio but skip tiny layers
#         if isinstance(mod, nn.Linear):
#             if mod.in_features <= 64 or mod.out_features <= 64:
#                 continue
#             k_max = min(mod.in_features, mod.out_features)
#             rank = max(1, int(math.ceil(k_max * rank_ratio)))
#             print(f"Decomposing Linear {name}: ({mod.out_features}, {mod.in_features}) -> rank {rank}")
#             new_mod = decompose_linear_layer(mod, rank)
#             setattr(parent, last_name, new_mod)
#
#         # Decompose pointwise Conv1d (kernel=1)
#         elif isinstance(mod, nn.Conv1d) and (mod.kernel_size == (1,) or mod.kernel_size == 1):
#             in_ch = mod.in_channels
#             out_ch = mod.out_channels
#             k_max = min(in_ch, out_ch)
#             rank = max(1, int(math.ceil(k_max * rank_ratio)))
#             # avoid decomposing very small convs
#             if max(in_ch, out_ch) >= 64:
#                 print(f"Decomposing Conv1d (1x1) {name}: ({out_ch}, {in_ch}, 1) -> rank {rank}")
#                 new_mod = decompose_conv1d_pointwise(mod, rank)
#                 setattr(parent, last_name, new_mod)
#
#     return model_new
#
# print("=" * 50)
# print("low-rank decomposition starts (Linear & 1x1 conv)...")
# model_lr = low_rank_decompose_model(model, rank_ratio=0.25, linear_min_size=128)
# model_lr.to(device)
# model_lr.eval()
#
# # Evaluate embeddings post-decomposition
# with torch.inference_mode():
#     out_lr = model_lr(signal_tensor)
#     emb_lr = out_lr[0].cpu().detach().numpy()
# sim_lr = cosine_similarity(emb_orig, emb_lr)
# print(f"original_lowrank_平均余弦相似度: {sim_lr:.4f}")
#
# # Save and report sizes
# torch.save(model_lr.state_dict(), "../weights/papagei_lowrank.pt")
# size_original = os.path.getsize("../weights/papagei_s.pt") / (1024*1024)  # MB
# size_lr = os.path.getsize("../weights/papagei_lowrank.pt") / (1024*1024)  # MB
# print(f"压缩率: {(size_original-size_lr)/size_original * 100:.2f}%")
#
#
############# Palettization (Weight Clustering)
print("=" * 50)
print("Palettization (Weight Clustering) starts")


class CompressedModel:
    """压缩模型类 - 存储聚类索引和码本"""

    def __init__(self):
        self.compressed_layers = {}
        self.uncompressed_params = {}
        self.model_structure = None

    def add_compressed_layer(self, layer_name, indices, codebook, original_shape):
        """添加压缩层信息"""
        self.compressed_layers[layer_name] = {
            'indices': indices.astype(np.uint8),  # 假设<256个聚类
            'codebook': codebook.astype(np.float32),
            'shape': original_shape,
            'n_clusters': len(codebook)
        }

    def add_uncompressed_param(self, param_name, param_value):
        """添加未压缩参数（如bias）"""
        # 统一使用cpu().numpy()以便pickle
        self.uncompressed_params[param_name] = param_value.cpu().numpy()

    def save(self, filepath):
        """保存压缩模型"""
        data = {
            'compressed_layers': self.compressed_layers,
            'uncompressed_params': self.uncompressed_params,
            'model_structure': self.model_structure
        }
        # 使用 pickle.dump 保存
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath):
        """加载压缩模型"""
        # 使用 pickle.load 加载
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.compressed_layers = data['compressed_layers']
        self.uncompressed_params = data['uncompressed_params']
        self.model_structure = data.get('model_structure')
        return self

    def decompress_to_state_dict(self):
        """解压缩为PyTorch state_dict格式"""
        state_dict = {}

        # 解压缩权重
        for layer_name, layer_data in self.compressed_layers.items():
            indices = layer_data['indices']
            codebook = layer_data['codebook']
            shape = layer_data['shape']

            # 重建权重
            weights = codebook[indices].reshape(shape)
            state_dict[layer_name] = torch.tensor(weights, dtype=torch.float32)

        # 添加未压缩参数
        for param_name, param_value in self.uncompressed_params.items():
            state_dict[param_name] = torch.tensor(param_value, dtype=torch.float32)

        return state_dict


def compress_model_with_clustering(model, n_clusters=16, layers_to_compress=(nn.Linear, nn.Conv1d)):
    """使用权重聚类压缩模型，并返回真正压缩的格式"""
    compressed_model = CompressedModel()
    reconstruction_model = copy.deepcopy(model)

    print(f"开始压缩模型，使用 {n_clusters} 个聚类...")

    total_params_compressed = 0
    total_layers_compressed = 0

    for name, module in model.named_modules():
        if isinstance(module, layers_to_compress):
            W = module.weight.data.cpu().numpy()
            original_shape = W.shape
            W_flat = W.reshape(-1)

            print(f"压缩 {name}: 形状 {original_shape}")

            if len(np.unique(W_flat)) > n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=42)
                kmeans.fit(W_flat.reshape(-1, 1))

                cluster_centers = kmeans.cluster_centers_.squeeze()
                labels = kmeans.labels_

                compressed_model.add_compressed_layer(
                    f"{name}.weight",
                    labels,
                    cluster_centers,
                    original_shape
                )

                W_reconstructed = cluster_centers[labels].reshape(original_shape)
                reconstruction_model.get_submodule(name).weight.data = torch.tensor(W_reconstructed,
                                                                                    dtype=torch.float32)

                total_params_compressed += len(W_flat)
                total_layers_compressed += 1

                print(f"  原始唯一值: {len(np.unique(W_flat))}")
                print(f"  压缩到: {n_clusters} 个聚类中心")
                print(f"  重建误差 (MSE): {np.mean((W_flat - cluster_centers[labels]) ** 2):.8f}")
            else:
                print(f"  跳过 {name} (唯一值太少)")
                compressed_model.add_uncompressed_param(f"{name}.weight", module.weight.data)

            if module.bias is not None:
                compressed_model.add_uncompressed_param(f"{name}.bias", module.bias.data)

    for param_name, param in model.named_parameters():
        param_parts = param_name.split('.')
        module_name = '.'.join(param_parts[:-1])
        param_type = param_parts[-1]

        if f"{module_name}.{param_type}" not in compressed_model.compressed_layers and \
                param_name not in compressed_model.uncompressed_params:
            compressed_model.add_uncompressed_param(param_name, param.data)

    print(f"\n压缩完成:")
    print(f"压缩层数: {total_layers_compressed}")
    print(f"压缩参数数: {total_params_compressed:,}")

    return compressed_model, reconstruction_model


def save_as_pkl(model, filepath):
    """将模型状态字典保存为 .pkl 文件"""
    # 过滤掉非张量对象，只对张量调用 .cpu()
    state_dict_cpu = {
        k: v.cpu() for k, v in model.state_dict().items()
        if isinstance(v, torch.Tensor)
    }
    with open(filepath, 'wb') as f:
        pickle.dump(state_dict_cpu, f)


def load_from_pkl(model, filepath):
    """从 .pkl 文件加载状态字典到模型"""
    with open(filepath, 'rb') as f:
        state_dict_cpu = pickle.load(f)
    model.load_state_dict(state_dict_cpu)
    return model


def quick_compression_test(model, signal_tensor, emb_orig, n_clusters=16):
    """快速测试单个聚类数的压缩效果"""
    print(f"\n🎯 快速测试 {n_clusters} 聚类压缩")
    print("-" * 50)

    # 获取原始模型大小 (统一使用 .pkl 格式)
    original_pkl_path = "../weights/papagei_s.pkl"
    save_as_pkl(model, original_pkl_path)
    original_size = os.path.getsize(original_pkl_path) / (1024 * 1024)

    # 执行压缩
    compressed_model, reconstructed_model = compress_model_with_clustering(
        model, n_clusters=n_clusters
    )

    # 保存为 .pkl 文件
    compressed_path = f"../weights/papagei_palettized_{n_clusters}.pkl"
    compressed_model.save(compressed_path)
    compressed_size = os.path.getsize(compressed_path) / (1024 * 1024)

    # 测试性能
    with torch.inference_mode():
        outputs = reconstructed_model(signal_tensor)
        emb_compressed = outputs[0].cpu().detach().numpy()
    similarity = cosine_similarity(emb_orig, emb_compressed)
    compression_ratio = (original_size - compressed_size) / original_size

    print(f"\n📊 压缩结果:")
    print(f"  原始文件: {original_size:.2f} MB")
    print(f"  压缩文件: {compressed_size:.2f} MB")
    print(f"  真实压缩率: {compression_ratio * 100:.1f}%")
    print(f"  相似度: {similarity:.4f}")

    return {
        'compressed_model': compressed_model,
        'reconstructed_model': reconstructed_model,
        'original_size_mb': original_size,
        'compressed_size_mb': compressed_size,
        'compression_ratio': compression_ratio,
        'similarity': similarity,
        'compressed_path': compressed_path
    }


def compare_all_methods(model, signal_tensor, emb_orig):
    """比较所有压缩方法 - 全部保存为 .pkl 格式"""
    print(f"\n{'=' * 80}")
    print("🔍 所有压缩方法对比 (公平比较 - 全部 .pkl 格式)")
    print(f"{'=' * 80}")

    results = {}
    original_pkl_path = "../weights/papagei_s.pkl"
    save_as_pkl(model, original_pkl_path)
    original_size = os.path.getsize(original_pkl_path) / (1024 * 1024)

    # 1. INT8 量化
    print("\n1. INT8 量化")
    print("-" * 30)
    torch.backends.quantized.engine = "qnnpack"
    model_int8 = quantize_dynamic(model, {torch.nn.Linear, torch.nn.Conv1d}, dtype=torch.qint8)

    int8_pkl_path = "../weights/papagei_int8.pkl"
    save_as_pkl(model_int8, int8_pkl_path)
    int8_size = os.path.getsize(int8_pkl_path) / (1024 * 1024)

    with torch.inference_mode():
        outputs_int8 = model_int8(signal_tensor)
        emb_int8 = outputs_int8[0].cpu().detach().numpy()
    sim_int8 = cosine_similarity(emb_orig, emb_int8)
    int8_compression = (original_size - int8_size) / original_size
    print(f"压缩率: {int8_compression * 100:.1f}%, 相似度: {sim_int8:.4f}")

    results['INT8量化'] = {
        'compression_ratio': int8_compression,
        'similarity': sim_int8,
        'size_mb': int8_size
    }

    # 2. 结构性剪枝
    print("\n2. 结构性剪枝")
    print("-" * 30)

    def structured_prune_model(model_to_prune, amount=0.1):
        pruned_model_copy = copy.deepcopy(model_to_prune)
        for name, module in pruned_model_copy.named_modules():
            if isinstance(module, torch.nn.Conv1d):
                prune.ln_structured(module, name="weight", amount=amount, n=1, dim=0)
                prune.remove(module, "weight")
        return pruned_model_copy

    pruned_model = structured_prune_model(model, amount=0.1)
    pruned_pkl_path = "../weights/papagei_struct_pruned.pkl"
    save_as_pkl(pruned_model, pruned_pkl_path)
    pruned_size = os.path.getsize(pruned_pkl_path) / (1024 * 1024)

    with torch.inference_mode():
        outputs_pruned = pruned_model(signal_tensor)
        emb_pruned = outputs_pruned[0].cpu().detach().numpy()
    sim_pruned = cosine_similarity(emb_orig, emb_pruned)
    pruned_compression = (original_size - pruned_size) / original_size
    print(f"压缩率: {pruned_compression * 100:.1f}%, 相似度: {sim_pruned:.4f}")

    results['结构性剪枝'] = {
        'compression_ratio': pruned_compression,
        'similarity': sim_pruned,
        'size_mb': pruned_size
    }

    # 3. 非结构化剪枝
    print("\n3. 非结构化剪枝")
    print("-" * 30)

    def unstructured_prune_model(model_to_prune, amount=0.1):
        unstructured_pruned_model_copy = copy.deepcopy(model_to_prune)
        parameters_to_prune = []
        for name, module in unstructured_pruned_model_copy.named_modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
                if hasattr(module, 'bias') and module.bias is not None:
                    parameters_to_prune.append((module, 'bias'))
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        return unstructured_pruned_model_copy

    unstruct_pruned_model = unstructured_prune_model(model, amount=0.01)
    unstruct_pruned_pkl_path = "../weights/papagei_unstruct_pruned.pkl"
    save_as_pkl(unstruct_pruned_model, unstruct_pruned_pkl_path)
    unstruct_pruned_size = os.path.getsize(unstruct_pruned_pkl_path) / (1024 * 1024)
    with torch.inference_mode():
        outputs_unstruct = unstruct_pruned_model(signal_tensor)
        emb_unstruct = outputs_unstruct[0].cpu().detach().numpy()
    sim_unstruct = cosine_similarity(emb_orig, emb_unstruct)
    unstruct_compression = (original_size - unstruct_pruned_size) / original_size
    print(f"压缩率: {unstruct_compression * 100:.1f}%, 相似度: {sim_unstruct:.4f}")

    results['非结构化剪枝'] = {
        'compression_ratio': unstruct_compression,
        'similarity': sim_unstruct,
        'size_mb': unstruct_pruned_size
    }

    # 4. 低秩分解
    print("\n4. 低秩分解")
    print("-" * 30)

    def decompose_linear_layer(linear: nn.Linear, rank: int):
        W = linear.weight.data.clone()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh[:rank, :]
        first = nn.Linear(linear.in_features, rank, bias=False)
        second = nn.Linear(rank, linear.out_features, bias=True)
        first.weight.data = Vh_r.clone()
        second.weight.data = (U_r * S_r.unsqueeze(0)).clone()
        if linear.bias is not None:
            second.bias.data = linear.bias.data.clone()
        else:
            second.bias.data.zero_()
        return nn.Sequential(first, second)

    def decompose_conv1d_pointwise(conv: nn.Conv1d, rank: int):
        assert conv.kernel_size == (1,) or conv.kernel_size == 1, "只处理 pointwise conv (kernel=1)"
        W = conv.weight.data.clone()
        out_ch, in_ch = W.shape[0], W.shape[1]
        W2d = W.view(out_ch, in_ch)
        U, S, Vh = torch.linalg.svd(W2d, full_matrices=False)
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh[:rank, :]
        conv1 = nn.Conv1d(in_channels=in_ch, out_channels=rank, kernel_size=1, bias=False)
        conv2 = nn.Conv1d(in_channels=rank, out_channels=out_ch, kernel_size=1, bias=(conv.bias is not None))
        conv1.weight.data = Vh_r.clone().view(rank, in_ch, 1)
        conv2.weight.data = (U_r * S_r.unsqueeze(0)).clone().view(out_ch, rank, 1)
        if conv.bias is not None:
            conv2.bias.data = conv.bias.data.clone()
        return nn.Sequential(conv1, conv2)

    def low_rank_decompose_model(model_to_decompose, rank_ratio=0.25):
        model_new = copy.deepcopy(model_to_decompose)
        for name, module in list(model_new.named_modules()):
            if name == "":
                continue
            path = name.split('.')
            parent = model_new
            for p in path[:-1]:
                parent = getattr(parent, p)
            last_name = path[-1]
            mod = getattr(parent, last_name)
            if isinstance(mod, nn.Linear):
                if mod.in_features <= 64 or mod.out_features <= 64:
                    continue
                k_max = min(mod.in_features, mod.out_features)
                rank = max(1, int(math.ceil(k_max * rank_ratio)))
                print(f"Decomposing Linear {name}: ({mod.out_features}, {mod.in_features}) -> rank {rank}")
                new_mod = decompose_linear_layer(mod, rank)
                setattr(parent, last_name, new_mod)
            elif isinstance(mod, nn.Conv1d) and (mod.kernel_size == (1,) or mod.kernel_size == 1):
                in_ch = mod.in_channels
                out_ch = mod.out_channels
                k_max = min(in_ch, out_ch)
                rank = max(1, int(math.ceil(k_max * rank_ratio)))
                if max(in_ch, out_ch) >= 64:
                    print(f"Decomposing Conv1d (1x1) {name}: ({out_ch}, {in_ch}, 1) -> rank {rank}")
                    new_mod = decompose_conv1d_pointwise(mod, rank)
                    setattr(parent, last_name, new_mod)
        return model_new

    model_lr = low_rank_decompose_model(model, rank_ratio=0.25)
    model_lr.to(device)
    model_lr.eval()
    lr_pkl_path = "../weights/papagei_lowrank.pkl"
    save_as_pkl(model_lr, lr_pkl_path)
    lr_size = os.path.getsize(lr_pkl_path) / (1024 * 1024)

    with torch.inference_mode():
        outputs_lr = model_lr(signal_tensor)
        emb_lr = outputs_lr[0].cpu().detach().numpy()
    sim_lr = cosine_similarity(emb_orig, emb_lr)
    lr_compression = (original_size - lr_size) / original_size
    print(f"压缩率: {lr_compression * 100:.1f}%, 相似度: {sim_lr:.4f}")

    results['低秩分解'] = {
        'compression_ratio': lr_compression,
        'similarity': sim_lr,
        'size_mb': lr_size
    }

    # 5. 权重聚类 (不同聚类数)
    cluster_options = [8, 16, 32]
    for n_clusters in cluster_options:
        print(f"\n5. 权重聚类 ({n_clusters}聚类)")
        print("-" * 30)

        compressed_model, reconstructed_model = compress_model_with_clustering(
            model, n_clusters=n_clusters
        )
        cluster_pkl_path = f"../weights/papagei_palettized_{n_clusters}.pkl"
        compressed_model.save(cluster_pkl_path)
        cluster_size = os.path.getsize(cluster_pkl_path) / (1024 * 1024)

        with torch.inference_mode():
            outputs_cluster = reconstructed_model(signal_tensor)
            emb_cluster = outputs_cluster[0].cpu().detach().numpy()
        sim_cluster = cosine_similarity(emb_orig, emb_cluster)
        cluster_compression = (original_size - cluster_size) / original_size
        print(f"压缩率: {cluster_compression * 100:.1f}%, 相似度: {sim_cluster:.4f}")

        results[f'聚类{n_clusters}'] = {
            'compression_ratio': cluster_compression,
            'similarity': sim_cluster,
            'size_mb': cluster_size
        }

    # 6. INT8 + 聚类32
    compressed_int8_model, reconstructed_int8_clustered = compress_model_with_clustering(
        model_int8, n_clusters=32
    )
    int8_clustered_path = "../weights/papagei_int8_cluster32.pkl"
    compressed_int8_model.save(int8_clustered_path)
    int8_clustered_size = os.path.getsize(int8_clustered_path) / (1024 * 1024)

    with torch.inference_mode():
        outputs_int8_clustered = reconstructed_int8_clustered(signal_tensor)
        emb_int8_clustered = outputs_int8_clustered[0].cpu().detach().numpy()
    sim_int8_clustered = cosine_similarity(emb_orig, emb_int8_clustered)
    int8_clustered_compression = (original_size - int8_clustered_size) / original_size
    print(f"压缩率: {int8_clustered_compression * 100:.1f}%, 相似度: {sim_int8_clustered:.4f}")

    results['INT8+聚类32'] = {
        'compression_ratio': int8_clustered_compression,
        'similarity': sim_int8_clustered,
        'size_mb': int8_clustered_size
    }

    print(f"\n{'=' * 60}")
    print("📊 压缩方法总结 (全部 .pkl 格式)")
    print(f"{'=' * 60}")
    print(f"{'方法':<12} {'压缩率':<8} {'相似度':<8} {'大小(MB)':<10}")
    print("-" * 45)

    for method, result in results.items():
        print(f"{method:<12} {result['compression_ratio'] * 100:<7.1f}% "
              f"{result['similarity']:<8.4f} {result['size_mb']:<10.2f}")

    return results


# 将原始模型也保存为 .pkl 文件，以便进行公平比较
print("准备原始模型 .pkl 文件...")
original_pkl_path = "../weights/papagei_s.pkl"
save_as_pkl(model, original_pkl_path)
print("准备完成！")

# Option 1: Quick test with 16 clusters
print("🚀 选项1: 快速测试 16 聚类")
result_32 = quick_compression_test(model, signal_tensor, emb_orig, n_clusters=32)

# Option 2: Compare all methods (takes longer)
print("\n🚀 选项2: 完整对比所有方法")
all_results = compare_all_methods(model, signal_tensor, emb_orig)

print(f"\n{'=' * 80}")
print("🎉 压缩测试完成!")
print(f"{'=' * 80}")
print("主要发现:")
print(f"- 真正的权重聚类压缩可以达到 {result_32['compression_ratio'] * 100:.1f}% 压缩率")
print(f"- 性能保持在 {result_32['similarity']:.4f} 相似度")
print(f"- 压缩文件已保存到: {result_32['compressed_path']}")

# Option 3: Load and test the compressed model
print(f"\n🔄 测试加载压缩模型...")
try:
    compressed_model = CompressedModel().load(result_32['compressed_path'])
    state_dict = compressed_model.decompress_to_state_dict()

    # Create new model instance and load decompressed weights
    decompressed_model = copy.deepcopy(model)
    decompressed_model.load_state_dict(state_dict, strict=False)
    decompressed_model.eval()

    # Test decompressed model
    with torch.inference_mode():
        outputs_decompressed = decompressed_model(signal_tensor)
        emb_decompressed = outputs_decompressed[0].cpu().detach().numpy()

    similarity_decompressed = cosine_similarity(emb_orig, emb_decompressed)
    print(f"✅ 解压缩模型性能: 相似度 {similarity_decompressed:.4f}")

except Exception as e:
    print(f"❌ 加载压缩模型失败: {e}")

print(f"\n{'=' * 50}")
print("Demo完成! 🎊")
print(f"{'=' * 50}")