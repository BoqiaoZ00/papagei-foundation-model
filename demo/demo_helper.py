import torch
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.stats import wasserstein_distance
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# ---------- 1. Embedding space consistency ----------

def evaluate_embedding_consistency(emb_orig, emb_compressed):
    results = {}

    # Norm statistics
    norm_orig = np.linalg.norm(emb_orig, axis=1)
    norm_comp = np.linalg.norm(emb_compressed, axis=1)

    results["mean_norm_shift"] = abs(norm_orig.mean() - norm_comp.mean())
    results["std_norm_shift"] = abs(norm_orig.std() - norm_comp.std())

    # Distribution similarity (Wasserstein distance)
    results["wasserstein_norm"] = wasserstein_distance(norm_orig, norm_comp)

    # Pairwise cosine similarity matrix
    cos_sim_matrix_orig = 1 - pairwise_distances(emb_orig, metric="cosine")
    cos_sim_matrix_comp = 1 - pairwise_distances(emb_compressed, metric="cosine")

    # Flatten upper triangle (to avoid redundancy)
    idx = np.triu_indices_from(cos_sim_matrix_orig, k=1)
    sims_orig = cos_sim_matrix_orig[idx]
    sims_comp = cos_sim_matrix_comp[idx]

    # Correlation between similarity structures
    corr = np.corrcoef(sims_orig, sims_comp)[0, 1]
    results["pairwise_similarity_corr"] = corr

    return results


# ---------- 2. Signal-level reconstruction ----------

class SimpleDecoder(torch.nn.Module):
    """Tiny decoder: embedding -> 1D signal reconstruction"""
    def __init__(self, emb_dim=512, signal_len=1250):
        super().__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, signal_len)
        )

    def forward(self, x):
        return self.decoder(x)


def evaluate_signal_reconstruction(embeddings, embeddings_comp, signal_tensor, device="cpu"):
    """
    embeddings: 原始模型 embedding (N, D)
    embeddings_comp: 压缩模型 embedding (N, D)
    signal_tensor: 原始输入信号 (N, 1, L)
    """
    N, _, signal_len = signal_tensor.shape
    emb_dim = embeddings.shape[1]

    # Train a tiny decoder to reconstruct signals from original embeddings
    decoder = SimpleDecoder(emb_dim=emb_dim, signal_len=signal_len).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    X = torch.tensor(embeddings, dtype=torch.float32, device=device)
    Y = signal_tensor.squeeze(1).to(device)  # (N, L)

    decoder.train()
    for _ in range(200):  # small training epochs
        optimizer.zero_grad()
        pred = decoder(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        optimizer.step()

    # Evaluate reconstruction from original and compressed embeddings
    decoder.eval()
    with torch.inference_mode():
        X_orig = torch.tensor(embeddings, dtype=torch.float32, device=device)
        X_comp = torch.tensor(embeddings_comp, dtype=torch.float32, device=device)

        rec_orig = decoder(X_orig).cpu().numpy()
        rec_comp = decoder(X_comp).cpu().numpy()
        true_signal = Y.cpu().numpy()

    # Metrics
    mse_orig = np.mean((rec_orig - true_signal) ** 2)
    mse_comp = np.mean((rec_comp - true_signal) ** 2)

    snr_orig = 10 * np.log10(np.sum(true_signal ** 2) / np.sum((true_signal - rec_orig) ** 2))
    snr_comp = 10 * np.log10(np.sum(true_signal ** 2) / np.sum((true_signal - rec_comp) ** 2))

    return {
        "mse_orig": mse_orig,
        "mse_comp": mse_comp,
        "snr_orig": snr_orig,
        "snr_comp": snr_comp,
    }

def test_reconstruction_decoder(emb_orig, emb_comp, signal_tensor, device="cpu"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Simple decoder (you can replace with your previous SimpleDecoder) ----
    class SimpleDecoder(nn.Module):
        def __init__(self, emb_dim=512, signal_len=1250, hidden=1024, dropout=0.0):
            super().__init__()
            layers = [
                nn.Linear(emb_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, signal_len)
            ]
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    # ---- utilities ----
    def to_tensor(x, dtype=torch.float32, device=device):
        return torch.tensor(x, dtype=dtype, device=device)

    def train_decoder(decoder, X_train, Y_train, X_val, Y_val,
                      epochs=200, lr=1e-3, weight_decay=0.0, batch_size=64, patience=20, verbose=False):
        decoder.to(device)
        opt = optim.Adam(decoder.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        best_val = float('inf')
        best_state = None
        wait = 0

        # convert to tensors
        X_train_t = to_tensor(X_train)
        Y_train_t = to_tensor(Y_train)
        X_val_t = to_tensor(X_val)
        Y_val_t = to_tensor(Y_val)

        n = X_train.shape[0]
        for ep in range(epochs):
            decoder.train()
            perm = np.random.permutation(n)
            epoch_loss = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                xb = X_train_t[idx]
                yb = Y_train_t[idx]
                pred = decoder(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= n

            # val
            decoder.eval()
            with torch.inference_mode():
                val_pred = decoder(X_val_t)
                val_loss = loss_fn(val_pred, Y_val_t).item()

            if verbose and (ep % 10 == 0 or ep == epochs - 1):
                print(f"ep {ep:03d} train_loss={epoch_loss:.6f} val_loss={val_loss:.6f}")

            if val_loss < best_val - 1e-8:
                best_val = val_loss
                best_state = decoder.state_dict()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {ep}, best_val={best_val:.6f}")
                    break

        if best_state is not None:
            decoder.load_state_dict(best_state)
        return decoder, best_val

    def evaluate_decoder(decoder, X, Y):
        decoder.eval()
        X_t = to_tensor(X)
        with torch.inference_mode():
            pred = decoder(X_t).cpu().numpy()
        Y_np = Y.copy() if isinstance(Y, np.ndarray) else Y.cpu().numpy()
        mse = np.mean((pred - Y_np) ** 2)
        # SNR (per-sample average)
        eps = 1e-12
        signal_power = np.mean(Y_np ** 2)
        noise_power = np.mean((pred - Y_np) ** 2) + eps
        snr = 10 * np.log10(signal_power / noise_power)
        return mse, snr, pred

    # ---- Prepare data (train/val split) ----
    # emb_orig, emb_comp: numpy arrays (N, D)
    # signal_tensor: torch.Tensor (N,1,L)
    N = emb_orig.shape[0]
    assert emb_orig.shape == emb_comp.shape
    signal_np = signal_tensor.squeeze(1).cpu().numpy()  # (N, L)

    # split into train/val/test indexes
    idx = np.arange(N)
    idx_train, idx_temp = train_test_split(idx, test_size=0.4, random_state=42)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=43)

    # Datasets
    X_orig_train, X_orig_val, X_orig_test = emb_orig[idx_train], emb_orig[idx_val], emb_orig[idx_test]
    X_comp_train, X_comp_val, X_comp_test = emb_comp[idx_train], emb_comp[idx_val], emb_comp[idx_test]
    Y_train, Y_val, Y_test = signal_np[idx_train], signal_np[idx_val], signal_np[idx_test]

    print("Train/Val/Test sizes:", len(idx_train), len(idx_val), len(idx_test))

    # ---- 3 decoders: train on orig, on comp, on mixed ----
    emb_dim = emb_orig.shape[1]
    signal_len = signal_np.shape[1]

    # hyperparams (you can tune)
    epochs = 500
    lr = 1e-3
    weight_decay = 1e-5
    patience = 20
    batch_size = 64
    dropout = 0.1

    # 1) decoder_orig (train on emb_orig)
    dec_orig = SimpleDecoder(emb_dim, signal_len, hidden=1024, dropout=dropout)
    dec_orig, best_val_orig = train_decoder(dec_orig,
                                            X_orig_train, Y_train, X_orig_val, Y_val,
                                            epochs=epochs, lr=lr, weight_decay=weight_decay, batch_size=batch_size,
                                            patience=patience, verbose=True)

    # 2) decoder_comp (train on emb_comp)
    dec_comp = SimpleDecoder(emb_dim, signal_len, hidden=1024, dropout=dropout)
    dec_comp, best_val_comp = train_decoder(dec_comp,
                                            X_comp_train, Y_train, X_comp_val, Y_val,
                                            epochs=epochs, lr=lr, weight_decay=weight_decay, batch_size=batch_size,
                                            patience=patience, verbose=True)

    # 3) decoder_mix (train on concatenated inputs)
    X_mix_train = np.vstack([X_orig_train, X_comp_train])
    Y_mix_train = np.vstack([Y_train, Y_train])
    # create validation mix as well
    X_mix_val = np.vstack([X_orig_val, X_comp_val])
    Y_mix_val = np.vstack([Y_val, Y_val])

    dec_mix = SimpleDecoder(emb_dim, signal_len, hidden=1024, dropout=dropout)
    dec_mix, best_val_mix = train_decoder(dec_mix,
                                          X_mix_train, Y_mix_train, X_mix_val, Y_mix_val,
                                          epochs=epochs, lr=lr, weight_decay=weight_decay, batch_size=batch_size,
                                          patience=patience, verbose=True)

    # ---- Evaluate cross-wise on test set ----
    m_orig_on_orig = evaluate_decoder(dec_orig, X_orig_test, Y_test)  # expect best
    m_orig_on_comp = evaluate_decoder(dec_orig, X_comp_test, Y_test)
    m_comp_on_comp = evaluate_decoder(dec_comp, X_comp_test, Y_test)  # expect best for comp
    m_comp_on_orig = evaluate_decoder(dec_comp, X_orig_test, Y_test)
    m_mix_on_orig = evaluate_decoder(dec_mix, X_orig_test, Y_test)
    m_mix_on_comp = evaluate_decoder(dec_mix, X_comp_test, Y_test)

    print("\n=== Cross-eval results (MSE, SNR) ===")
    print("dec_orig on orig:", m_orig_on_orig[:2])
    print("dec_orig on comp:", m_orig_on_comp[:2])
    print("dec_comp on comp:", m_comp_on_comp[:2])
    print("dec_comp on orig:", m_comp_on_orig[:2])
    print("dec_mix on orig:", m_mix_on_orig[:2])
    print("dec_mix on comp:", m_mix_on_comp[:2])

    # Optional: visualize one test sample reconstructions
    i = 0
    _, _, rec_orig = evaluate_decoder(dec_orig, X_orig_test[:1], Y_test[:1])
    _, _, rec_comp = evaluate_decoder(dec_comp, X_comp_test[:1], Y_test[:1])
    _, _, rec_mix_o = evaluate_decoder(dec_mix, X_orig_test[:1], Y_test[:1])
    _, _, rec_mix_c = evaluate_decoder(dec_mix, X_comp_test[:1], Y_test[:1])

    true = Y_test[:1][0]
    plt.figure(figsize=(10, 4))
    plt.plot(true, label='true')
    plt.plot(rec_orig[0], label='rec_orig')
    plt.plot(rec_comp[0], label='rec_comp')
    plt.plot(rec_mix_c[0], label='rec_mix_comp')
    plt.legend();
    plt.show()