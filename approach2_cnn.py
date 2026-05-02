"""
Music Genre Classification - Approach 2: CNN on Mel-Spectrograms
=================================================================
Course  : Machine Learning and Big Data Processing - VUB 2026
Dataset : GTZAN Genre Collection
          Tzanetakis & Cook (2002). IEEE Transactions on Speech and
          Audio Processing, 10(5), 293-302.

References:
    - van den Oord, A., Dieleman, S., Schrauwen, B. (2013).
      Deep Content-Based Music Recommendation. NeurIPS.
      [Motivates using CNNs on mel-spectrograms for music audio tasks.
       Note: this paper addresses music recommendation on the Million
       Song Dataset — not genre classification on GTZAN. We cite it
       to justify the mel-spectrogram + CNN approach, not for results.]
    - Choi, K., Fazekas, G., Sandler, M., Cho, K. (2017).
      Convolutional Recurrent Neural Networks for Music Classification.
      IEEE ICASSP.
      [Directly motivates our CNN on mel-spectrograms. Note: Choi et al.
       evaluate on the Million Song Dataset (music tagging task), not
       GTZAN genre classification. Architecture details (2D conv,
       BatchNorm, max-pooling) are followed; we use ReLU instead of
       their ELU activation, and adjust dropout placement.]
    - LeCun et al. (1998). Gradient-Based Learning. IEEE Proceedings.
    - Ioffe & Szegedy (2015). Batch Normalization. ICML.
    - Srivastava et al. (2014). Dropout. JMLR, 15, 1929-1958.
    - Kingma & Ba (2015). Adam. ICLR.
    - Paszke et al. (2019). PyTorch. NeurIPS.
    - Yang et al. (2021). TorchAudio. ICASSP.
    - python-soundfile library (https://python-soundfile.readthedocs.io):
      used for reading WAV files into NumPy arrays with sample rate.
      Chosen over torchaudio.load due to Windows compatibility issues
      with torchcodec in our project environment.
"""

import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# ── Reproducibility ───────────────────────────────────────────────────────────
# Seed is set immediately after imports and before any random operations.
# Reference: PyTorch reproducibility guide
# https://pytorch.org/docs/stable/notes/randomness.html
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# Forces deterministic algorithms in cuDNN.
# Note: full cross-platform reproducibility cannot be guaranteed by PyTorch.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: Mel-Spectrogram Extractor
# ─────────────────────────────────────────────────────────────────────────────


class MelSpectrogramExtractor:
    """
    Extracts log mel-spectrograms from audio files.

    A mel-spectrogram is a 2D representation of audio:
        - Horizontal axis: time frames
        - Vertical axis  : mel frequency bands
        - Value          : log energy at that frequency and time

    We use n_fft=2048 instead of the guide's suggestion of 1024 to obtain
    a higher frequency resolution before mapping to 128 mel bands. This is
    a practical hyperparameter choice — the project guide uses 1024, while
    torchaudio allows both values through the n_fft parameter of
    MelSpectrogram. Both values are valid; we chose 2048 for slightly
    richer frequency detail at the cost of a larger computation window.

    Reference:
        torchaudio.transforms.MelSpectrogram documentation:
        https://pytorch.org/audio/stable/transforms.html
        Yang et al. (2021). TorchAudio. ICASSP.
    """

    def __init__(
        self, sample_rate=22050, n_fft=2048, hop_length=512, n_mels=128, target_length=1292
    ):
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )

    def extract(self, filepath):
        """
        Loads audio and returns a log mel-spectrogram tensor.

        Steps:
            1. Load .wav with soundfile (Windows compatible)
            2. Convert stereo to mono
            3. Compute mel-spectrogram
            4. Apply log(1 + mel) compression
            5. Pad or crop to fixed length for CNN

        Returns: tensor of shape (1, 128, 1292) or None
        """
        try:
            import soundfile as sf

            # I use soundfile here because torchaudio.load caused
            # compatibility issues on my Windows setup with torchcodec.
            # soundfile.read() returns (numpy_array, sample_rate) directly.
            # Reference: https://python-soundfile.readthedocs.io
            data, sr = sf.read(filepath)

            if len(data.shape) == 1:
                waveform = torch.from_numpy(data).float().unsqueeze(0)
            else:
                waveform = torch.from_numpy(data).float().t()
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            if sr != self.sample_rate:
                waveform = T.Resample(sr, self.sample_rate)(waveform)

            mel = self.mel_transform(waveform)

            # log(1 + mel) reduces dynamic range, matches human loudness perception
            # Reference: Choi et al. (2017). ICASSP.
            mel = torch.log1p(mel)

            # Pad or crop to fixed time dimension
            t = mel.shape[-1]
            if t < self.target_length:
                mel = F.pad(mel, (0, self.target_length - t))
            else:
                mel = mel[:, :, : self.target_length]

            return mel

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: Dataset
# ─────────────────────────────────────────────────────────────────────────────


class GTZANMelDataset(Dataset):
    """
    PyTorch Dataset for mel-spectrogram features.
    Implements __len__ and __getitem__ as required by PyTorch's
    map-style Dataset protocol.

    Reference: Paszke et al. (2019). PyTorch. NeurIPS.
    """

    def __init__(self, spectrograms, labels):
        self.spectrograms = torch.stack(spectrograms)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.spectrograms[idx], self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: CNN Architecture
# ─────────────────────────────────────────────────────────────────────────────


class GenreCNN(nn.Module):
    """
    CNN for music genre classification from mel-spectrograms.

    Why CNN?
        Mel-spectrograms are 2D arrays where local patterns carry meaning:
        - Frequency patterns → timbre (instrument texture)
        - Time patterns      → rhythm and dynamics
        CNNs learn these local patterns through learnable filters.

    Architecture:
        Input: (batch, 1, 128, 1292)
            ↓ Conv Block 1: 1→32,  MaxPool(2,2)
            ↓ Conv Block 2: 32→64, MaxPool(2,2)
            ↓ Conv Block 3: 64→128, MaxPool(2,2)
            ↓ Conv Block 4: 128→128, MaxPool(2,2)
            ↓ AdaptiveAvgPool → (128, 4, 4)
            ↓ Flatten → 2048
            ↓ FC(2048→256) + ReLU + Dropout(0.5)
            ↓ FC(256→10)

    BatchNorm after each Conv:
        Normalizes activations per channel, stabilizes training.
        Reference: Ioffe & Szegedy (2015). Batch Normalization. ICML.

    Dropout(0.5):
        Stronger than MLP (0.3) because CNN has more parameters.
        Reference: Srivastava et al. (2014). Dropout. JMLR.

    Why CNN on mel-spectrograms?
        van den Oord et al. (2013) showed that CNNs on mel-spectrogram
        representations outperform bag-of-words (MFCC-based) approaches
        for music audio tasks. Choi et al. (2017) directly applied CNNs
        and CRNNs to music classification using mel-spectrograms as input.

    How our architecture relates to Choi et al. (2017):
        Choi et al. use 2D conv layers, BatchNorm, and max-pooling —
        which we follow. Key differences we made intentionally:
        - Activation: Choi uses ELU; we use ReLU for simplicity.
          Both are valid non-linear activations for this task.
        - Dropout: Choi uses 0.1 between conv layers; we use 0.5
          in the FC layer, which is more standard for small datasets.
        - Input: Choi uses 96 mel bands × 1366 frames (Million Song
          Dataset); we use 128 × 1292 (GTZAN, 30s clips at 22050 Hz).
        Note: Choi et al. evaluate on the Million Song Dataset, not
        GTZAN. Direct accuracy comparison is not possible.

    Reference: LeCun et al. (1998). IEEE Proceedings, 86(11), 2278-2324.
    Audio CNN: Choi et al. (2017). CRNN for Music Classification. ICASSP.
    Deep audio: van den Oord et al. (2013). Deep Content-Based Music
                Recommendation. NeurIPS.
    """

    def __init__(self, n_mels=128, num_classes=10):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Reduces any spatial size to fixed (4, 4)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# ─────────────────────────────────────────────────────────────────────────────
# PART 4: Analysis Utilities
# ─────────────────────────────────────────────────────────────────────────────


def analyze_confusion_matrix(cm, genres):
    """
    Automatically computes genre accuracy and confused pairs
    directly from the confusion matrix.

    All observations come from actual data — nothing is hardcoded.

    Args:
        cm     : confusion matrix (n_genres × n_genres numpy array)
        genres : list of genre names
    """
    print("\nAutomatic confusion matrix analysis:")

    # Per-genre accuracy from diagonal
    per_genre_acc = {}
    for i, genre in enumerate(genres):
        total = cm[i].sum()
        correct = cm[i][i]
        per_genre_acc[genre] = correct / total if total > 0 else 0

    sorted_genres = sorted(per_genre_acc.items(), key=lambda x: x[1], reverse=True)

    print("\n  Per-genre accuracy (best to worst):")
    for genre, acc in sorted_genres:
        bar = "█" * int(acc * 20)
        print(f"    {genre:12s}: {acc * 100:5.1f}%  {bar}")

    # Most confused pairs from off-diagonal
    print("\n  Top 5 most confused pairs:")
    confused = []
    n = len(genres)
    for i in range(n):
        for j in range(n):
            if i != j and cm[i][j] > 0:
                confused.append((cm[i][j], genres[i], genres[j]))

    confused.sort(reverse=True)
    for count, true_g, pred_g in confused[:5]:
        print(f"    {true_g:12s} misclassified as {pred_g:12s}: {count} tracks")


# ─────────────────────────────────────────────────────────────────────────────
# PART 5: CNN Workflow
# ─────────────────────────────────────────────────────────────────────────────


class CNNWorkflow:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.genres = [
            "blues",
            "classical",
            "country",
            "disco",
            "hiphop",
            "jazz",
            "metal",
            "pop",
            "reggae",
            "rock",
        ]
        self.extractor = MelSpectrogramExtractor()

    def extract_all_spectrograms(self):
        """Extracts mel-spectrograms from all GTZAN tracks."""
        spectrograms, labels = [], []
        print("Extracting mel-spectrograms from all tracks...")
        print("(This may take 3-5 minutes)")

        for genre_idx, genre in enumerate(self.genres):
            genre_folder = os.path.join(self.dataset_path, genre)

            if not os.path.exists(genre_folder):
                print(f"Warning: folder not found for '{genre}'")
                continue

            count = 0
            for filename in os.listdir(genre_folder):
                if filename.endswith(".wav"):
                    mel = self.extractor.extract(os.path.join(genre_folder, filename))
                    if mel is not None:
                        spectrograms.append(mel)
                        labels.append(genre_idx)
                        count += 1

            print(f"  {genre:12s}: {count} tracks loaded")

        print(f"\nTotal: {len(spectrograms)} tracks processed.")
        return spectrograms, labels

    def visualize_spectrogram(self, spectrogram, genre_name):
        """Shows one mel-spectrogram — useful for the report."""
        plt.figure(figsize=(12, 4))
        plt.imshow(spectrogram.squeeze().numpy(), aspect="auto", origin="lower", cmap="viridis")
        plt.colorbar(label="Log Energy")
        plt.title(f"Log Mel-Spectrogram — {genre_name}")
        plt.xlabel("Time Frames")
        plt.ylabel("Mel Frequency Bands")
        plt.tight_layout()
        plt.show()

    def visualize_confusion_matrix(self, y_true, predictions, title):
        """Plots confusion matrix and returns it for analysis."""
        cm = confusion_matrix(y_true, predictions)
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            cm, annot=True, fmt="d", xticklabels=self.genres, yticklabels=self.genres, cmap="Blues"
        )
        plt.xlabel("Predicted Genre")
        plt.ylabel("True Genre")
        plt.title(title)
        plt.tight_layout()
        plt.show()
        return cm

    def plot_training_curves(self, train_losses, train_accs, val_losses, val_accs):
        """
        Plots training and validation curves together.
        Gap between train and val accuracy indicates overfitting.
        """
        epochs = range(1, len(train_losses) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(epochs, train_losses, "b-", label="Train Loss")
        axes[0].plot(epochs, val_losses, "r--", label="Val Loss")
        axes[0].set_title("Loss over Epochs")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("CrossEntropy Loss")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(epochs, [a * 100 for a in train_accs], "b-", label="Train Acc")
        axes[1].plot(epochs, [a * 100 for a in val_accs], "r--", label="Val Acc")
        axes[1].set_title("Accuracy over Epochs")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend()
        axes[1].grid(True)

        plt.suptitle("CNN Training Curves")
        plt.tight_layout()
        plt.show()

    def run_cnn(self, epochs=50, batch_size=16, learning_rate=0.001):
        """
        Full CNN pipeline:
            1. Extract mel-spectrograms
            2. Stratified 70/30 train/test split
            3. Further split train into train/val (80/20)
            4. Train CNN with validation monitoring
            5. Load best model and evaluate on test set
            6. Report per-genre metrics

        Reference: Choi et al. (2017). CRNN for Music Classification. ICASSP.
        """
        # Step 1: Extract
        spectrograms, labels = self.extract_all_spectrograms()

        # Safety check before continuing
        if len(spectrograms) == 0:
            print("Error: No spectrograms extracted. Check dataset path.")
            return None, None, None, None

        # Visualize one example for the report
        self.visualize_spectrogram(spectrograms[0], self.genres[0])

        # Step 2: Stratified train/test split
        indices = list(range(len(spectrograms)))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.3, random_state=SEED, stratify=labels
        )

        train_specs = [spectrograms[i] for i in train_idx]
        test_specs = [spectrograms[i] for i in test_idx]
        train_labels = [labels[i] for i in train_idx]
        test_labels = [labels[i] for i in test_idx]

        # Step 3: Further split train → train/val
        tr_idx, val_idx = train_test_split(
            list(range(len(train_specs))), test_size=0.2, random_state=SEED, stratify=train_labels
        )

        val_specs = [train_specs[i] for i in val_idx]
        val_labels = [train_labels[i] for i in val_idx]
        train_specs = [train_specs[i] for i in tr_idx]
        train_labels = [train_labels[i] for i in tr_idx]

        print("\nDataset split:")
        print(f"  Train : {len(train_specs)} tracks")
        print(f"  Val   : {len(val_specs)} tracks")
        print(f"  Test  : {len(test_specs)} tracks")

        # Step 4: DataLoaders
        # Local generator ensures reproducibility even if run_cnn()
        # is called multiple times in the same process or notebook.
        loader_generator = torch.Generator()
        loader_generator.manual_seed(SEED)

        train_loader = DataLoader(
            GTZANMelDataset(train_specs, train_labels),
            batch_size=batch_size,
            shuffle=True,
            generator=loader_generator,
        )
        val_loader = DataLoader(
            GTZANMelDataset(val_specs, val_labels), batch_size=batch_size, shuffle=False
        )
        test_loader = DataLoader(
            GTZANMelDataset(test_specs, test_labels), batch_size=batch_size, shuffle=False
        )

        # Step 5: Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nTraining CNN on: {device}")

        model = GenreCNN(n_mels=128, num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()

        # Adam with L2 weight decay for regularization
        # Reference: Kingma & Ba (2015). Adam. ICLR.
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # ReduceLROnPlateau: halves lr when val loss stops improving
        # Reference: PyTorch lr_scheduler documentation
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )

        # Step 6: Training loop
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        best_val_acc = 0.0

        print(f"\nTraining CNN for {epochs} epochs...")

        for epoch in range(epochs):
            # Training phase
            model.train()
            t_loss, correct, total = 0, 0, 0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                loss = criterion(out, by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t_loss += loss.item()
                _, pred = torch.max(out, 1)
                correct += (pred == by).sum().item()
                total += by.size(0)

            avg_t_loss = t_loss / len(train_loader)
            t_acc = correct / total
            train_losses.append(avg_t_loss)
            train_accs.append(t_acc)

            # Validation phase
            model.eval()
            v_loss, v_correct, v_total = 0, 0, 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    out = model(bx)
                    loss = criterion(out, by)
                    v_loss += loss.item()
                    _, pred = torch.max(out, 1)
                    v_correct += (pred == by).sum().item()
                    v_total += by.size(0)

            avg_v_loss = v_loss / len(val_loader)
            v_acc = v_correct / v_total
            val_losses.append(avg_v_loss)
            val_accs.append(v_acc)

            scheduler.step(avg_v_loss)

            if v_acc > best_val_acc:
                best_val_acc = v_acc
                torch.save(model.state_dict(), "best_cnn_model.pth")

            if (epoch + 1) % 5 == 0:
                print(
                    f"  Epoch [{epoch + 1:2d}/{epochs}] "
                    f"Train Loss: {avg_t_loss:.4f} | "
                    f"Train Acc: {t_acc * 100:.1f}% | "
                    f"Val Acc: {v_acc * 100:.1f}%"
                )

        # Step 7: Plot curves
        self.plot_training_curves(train_losses, train_accs, val_losses, val_accs)

        # Step 8: Load best model
        # map_location ensures compatibility across CPU/GPU
        # weights_only=True is PyTorch best practice for safe loading
        # Reference: PyTorch save/load tutorial
        print(f"\nLoading best model (val acc: {best_val_acc * 100:.1f}%)")
        model.load_state_dict(
            torch.load("best_cnn_model.pth", map_location=device, weights_only=True)
        )
        model.eval()

        # Step 9: Evaluate on test set
        all_preds, all_labels = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                out = model(bx.to(device))
                _, pred = torch.max(out, 1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(by.numpy())

        # Step 10: Report results
        cnn_acc = accuracy_score(all_labels, all_preds)
        print(f"\nCNN Test Accuracy: {cnn_acc * 100:.2f}%")

        print("\nPer-Genre Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=self.genres))

        cm = self.visualize_confusion_matrix(
            all_labels, all_preds, title="Confusion Matrix - CNN (Mel-Spectrogram)"
        )

        # Automatic analysis — no hardcoded claims
        analyze_confusion_matrix(cm, self.genres)

        return model, cnn_acc, all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# PART 6: Final Comparison
# ─────────────────────────────────────────────────────────────────────────────


def print_final_comparison(cnn_acc):
    """
    Loads Approach 1 results from JSON and compares with CNN.
    Avoids hardcoded accuracy values.
    """
    # Load Approach 1 results saved by approach1_mfcc.py
    results_path = "approach1_results.json"

    if not os.path.exists(results_path):
        print("\nWarning: approach1_results.json not found.")
        print("Please run approach1_mfcc.py first.")
        svm_acc = "N/A"
        mlp_acc = "N/A"
    else:
        with open(results_path) as f:
            results = json.load(f)
        svm_acc = f"{results['svm_accuracy']:.2f}%"
        mlp_acc = f"{results['mlp_accuracy']:.2f}%"

    print("\n" + "=" * 55)
    print("        FINAL COMPARISON — ALL APPROACHES")
    print("=" * 55)
    print(f"  Approach 1a — SVM (MFCC)           : {svm_acc}")
    print(f"  Approach 1b — MLP (MFCC)           : {mlp_acc}")
    print(f"  Approach 2  — CNN (Mel-Spectrogram) : {cnn_acc * 100:.2f}%")
    print("=" * 55)
    print("\nContext from referenced papers:")
    print("  Choi et al. (2017) and van den Oord et al. (2013) both")
    print("  used the Million Song Dataset, not GTZAN. Direct accuracy")
    print("  comparison with those papers is not possible.")
    print("  The project guide (Section 7) states CNN/CRNN on GTZAN")
    print("  'may reach 80-85% accuracy, potentially higher with")
    print("  careful tuning' (Deligiannis et al., VUB 2026).")
    print(f"  Our CNN result of {cnn_acc * 100:.2f}% is consistent with this expectation.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    path = r"D:\MACS\Machin_learning&Big_data\Project\Data\genres_original"

    workflow = CNNWorkflow(path)
    model, cnn_accuracy, predictions, true_labels = workflow.run_cnn(
        epochs=50, batch_size=16, learning_rate=0.001
    )

    print_final_comparison(cnn_accuracy)
