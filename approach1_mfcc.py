"""
Music Genre Classification - Approach 1: MFCC + SVM / MLP
==========================================================
Course  : Machine Learning and Big Data Processing - VUB 2026
Dataset : GTZAN Genre Collection
          Tzanetakis & Cook (2002). IEEE Transactions on Speech and
          Audio Processing, 10(5), 293-302.

References:
    - Logan (2000). MFCC for Music Modeling. ISMIR.
    - Cortes & Vapnik (1995). Support-vector networks. Machine Learning.
    - Kingma & Ba (2015). Adam. ICLR.
    - Paszke et al. (2019). PyTorch. NeurIPS.
    - Pedregosa et al. (2011). Scikit-learn. JMLR, 12, 2825-2830.
    - Srivastava et al. (2014). Dropout. JMLR, 15, 1929-1958.
    - Goodfellow et al. (2016). Deep Learning. MIT Press.
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
import torchaudio
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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
# PART 1: Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────


class MusicGenreClassification:
    """
    Loads audio files and extracts MFCC features.

    MFCC pipeline:
        1. STFT  → frequency content per time frame
        2. Mel filterbank → match human hearing (mel scale)
        3. DCT   → compress into K coefficients

    We use n_mfcc=20 instead of the guide's suggestion of 13 because
    20 coefficients capture slightly more timbral detail, and the guide
    itself states "typically 13 or 20". Both are valid choices.

    Reference: Logan (2000). MFCC for Music Modeling. ISMIR.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.sample_rate = None
        self.MFCC = None
        self.waveform = None

    def extract_waveform(self):
        """
        Loads .wav file into a waveform tensor using soundfile.
        soundfile is used instead of torchaudio.load() to avoid
        torchcodec compatibility issues on Windows.

        GTZAN: sample_rate=22050 Hz, duration=30s
        → 30 × 22050 = 661,500 samples per track
        """
        try:
            import soundfile as sf

            # I use soundfile instead of torchaudio.load because torchaudio
            # caused torchcodec errors on my Windows machine. soundfile.read()
            # returns (numpy_array, sample_rate) and works reliably here.
            # Reference: https://python-soundfile.readthedocs.io
            data, sr = sf.read(self.filepath)
            self.sample_rate = sr

            if len(data.shape) == 1:
                self.waveform = torch.from_numpy(data).float().unsqueeze(0)
            else:
                self.waveform = torch.from_numpy(data).float().t()

            if self.waveform.shape[0] > 1:
                self.waveform = torch.mean(self.waveform, dim=0, keepdim=True)

            return self.waveform, self.sample_rate

        except Exception as e:
            print(f"Error loading {self.filepath}: {e}")
            return None, None

    def transform_MFCC(self):
        """
        Computes MFCC features from the waveform.

        Parameters match project guide recommendation with n_mfcc=20:
            n_mfcc      = 20   : number of cepstral coefficients
            n_fft       = 1024 : FFT window size (per project guide)
            hop_length  = 512  : step between frames
            n_mels      = 128  : mel frequency bands

        Output shape: (1, 20, ~1292 time frames)

        Reference:
            torchaudio.transforms.MFCC:
            Yang et al. (2021). TorchAudio. ICASSP.
        """
        if self.waveform is None:
            self.extract_waveform()

        if self.waveform is None:
            return None

        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            # I keep n_mfcc=20 because early experiments showed slightly
            # richer features than 13 coefficients, while still keeping
            # the vector small enough for SVM and MLP.
            # The project guide says "typically 13 or 20" — both are valid.
            n_mfcc=20,
            melkwargs={
                "n_fft": 1024,  # FFT window size (per project guide)
                "hop_length": 512,  # frame step
                "n_mels": 128,  # mel frequency bands
            },
        )
        self.MFCC = mfcc_transform(self.waveform)
        return self.MFCC

    def get_mfcc_mean(self):
        """
        Returns temporal mean of MFCC: shape (20,).

        We average over time to get a fixed-length vector per track.
        This is the standard approach for feeding MFCCs into SVM/MLP.

        Reference: Logan (2000). ISMIR.
        """
        if self.MFCC is None:
            self.transform_MFCC()

        if self.MFCC is None:
            return None

        return torch.mean(self.MFCC, dim=-1).squeeze().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────


class GTZANDataset(Dataset):
    """
    Wraps MFCC features and labels for PyTorch DataLoader.
    Implements __len__ and __getitem__ as required by PyTorch's
    map-style Dataset protocol.

    Reference: Paszke et al. (2019). PyTorch. NeurIPS.
    """

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: MLP Architecture
# ─────────────────────────────────────────────────────────────────────────────


class GenreMLP(nn.Module):
    """
    Simple MLP for genre classification from MFCC features.

    Architecture: 20 → 128 → 64 → 32 → 10

    Design choices:
        - ReLU: introduces non-linearity between layers
        - Dropout(0.3): prevents overfitting on small dataset
        - No softmax: CrossEntropyLoss applies it internally

    Reference: Goodfellow et al. (2016). Deep Learning. MIT Press. Ch.6.
    Dropout: Srivastava et al. (2014). JMLR, 15, 1929-1958.
    """

    def __init__(self, input_size=20, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        return self.output(x)


# ─────────────────────────────────────────────────────────────────────────────
# PART 4: Workflow
# ─────────────────────────────────────────────────────────────────────────────


class Workflow:
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

    def extract_all_features(self):
        """Loops through all genre folders and extracts MFCC mean vectors."""
        X, y = [], []
        print("Extracting MFCC features from all tracks...")

        for genre_idx, genre in enumerate(self.genres):
            genre_folder = os.path.join(self.dataset_path, genre)

            if not os.path.exists(genre_folder):
                print(f"Warning: folder not found for '{genre}'")
                continue

            for filename in os.listdir(genre_folder):
                if filename.endswith(".wav"):
                    filepath = os.path.join(genre_folder, filename)
                    processor = MusicGenreClassification(filepath)
                    mfcc_val = processor.get_mfcc_mean()

                    if mfcc_val is not None:
                        X.append(mfcc_val)
                        y.append(genre_idx)

        if len(X) == 0:
            print("Error: No data extracted. Check dataset path.")
            return None, None

        print(f"Extraction complete: {len(X)} tracks processed.")
        return np.array(X), np.array(y)

    def visualize_confusion_matrix(self, y_true, predictions, title):
        """Plots confusion matrix. Rows=true genre, Columns=predicted genre."""
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

    def analyze_confusion_matrix(self, cm, title):
        """
        Automatically finds easiest and hardest genres from confusion matrix.
        This avoids hardcoded claims — all observations come from actual data.
        """
        print(f"\nAutomatic analysis of {title}:")

        # Per-genre accuracy from diagonal
        per_genre_acc = {}
        for i, genre in enumerate(self.genres):
            total = cm[i].sum()
            correct = cm[i][i]
            per_genre_acc[genre] = correct / total if total > 0 else 0

        sorted_genres = sorted(per_genre_acc.items(), key=lambda x: x[1], reverse=True)

        print("  Genre accuracy (best to worst):")
        for genre, acc in sorted_genres:
            print(f"    {genre:12s}: {acc * 100:.1f}%")

        # Most confused pairs from off-diagonal
        print("\n  Most confused genre pairs:")
        confused_pairs = []
        n = len(self.genres)
        for i in range(n):
            for j in range(n):
                if i != j and cm[i][j] > 0:
                    confused_pairs.append((cm[i][j], self.genres[i], self.genres[j]))

        confused_pairs.sort(reverse=True)
        for count, true_g, pred_g in confused_pairs[:5]:
            print(f"    {true_g:12s} → {pred_g:12s}: {count} tracks misclassified")

    def run_svm(self, X_train, X_test, y_train, y_test, scaler):
        """
        Trains SVM with RBF kernel.

        sklearn.svm.SVC uses one-vs-one strategy for multiclass problems.
        For 10 genres this means C(10,2) = 45 binary classifiers.
        Final prediction is by majority vote.

        Reference: Cortes & Vapnik (1995). Machine Learning, 20(3), 273-297.
        sklearn docs: https://scikit-learn.org/stable/modules/svm.html

        Parameters:
            kernel='rbf'   : handles non-linearly separable genres
            C=10.0         : regularization — higher = smaller margin
            gamma='scale'  : 1 / (n_features * X.var())
        """
        X_train_sc = scaler.transform(X_train)
        X_test_sc = scaler.transform(X_test)

        print("\nTraining SVM model...")
        svm_model = SVC(kernel="rbf", C=10.0, gamma="scale")
        svm_model.fit(X_train_sc, y_train)

        predictions = svm_model.predict(X_test_sc)
        accuracy = accuracy_score(y_test, predictions)
        print(f"SVM Test Accuracy: {accuracy * 100:.2f}%")

        cm = self.visualize_confusion_matrix(y_test, predictions, "Confusion Matrix - SVM (MFCC)")
        self.analyze_confusion_matrix(cm, "SVM")
        return svm_model, predictions, accuracy

    def run_mlp(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
    ):
        """
        Trains MLP with Adam optimizer and CrossEntropyLoss.

        Key training concepts:
            - Epoch: full pass through training data
            - Mini-batch: subset processed at once (reduces memory use)
            - Loss: CrossEntropyLoss = LogSoftmax + NLLLoss
            - Adam: adaptive learning rate optimizer

        Reference: Kingma & Ba (2015). Adam. ICLR.
        """
        X_train_sc = scaler.transform(X_train)
        X_test_sc = scaler.transform(X_test)

        # Local generator ensures reproducibility even if run_mlp()
        # is called multiple times in the same process or notebook.
        loader_generator = torch.Generator()
        loader_generator.manual_seed(SEED)

        train_loader = DataLoader(
            GTZANDataset(X_train_sc, y_train),
            batch_size=batch_size,
            shuffle=True,
            generator=loader_generator,
        )
        test_loader = DataLoader(
            GTZANDataset(X_test_sc, y_test), batch_size=batch_size, shuffle=False
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GenreMLP(input_size=X_train.shape[1]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train_losses, train_accs = [], []
        print(f"\nTraining MLP on: {device}")

        for epoch in range(epochs):
            model.train()
            total_loss, correct, total = 0, 0, 0

            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                loss = criterion(out, by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                _, pred = torch.max(out, 1)
                correct += (pred == by).sum().item()
                total += by.size(0)

            avg_loss = total_loss / len(train_loader)
            acc = correct / total
            train_losses.append(avg_loss)
            train_accs.append(acc)

            if (epoch + 1) % 10 == 0:
                print(
                    f"  Epoch [{epoch + 1:3d}/{epochs}] "
                    f"Loss: {avg_loss:.4f} | "
                    f"Train Acc: {acc * 100:.2f}%"
                )

        # Plot training curves
        self._plot_curves(train_losses, train_accs)

        # Evaluate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                out = model(bx.to(device))
                _, pred = torch.max(out, 1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(by.numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        print(f"\nMLP Test Accuracy: {accuracy * 100:.2f}%")

        cm = self.visualize_confusion_matrix(all_labels, all_preds, "Confusion Matrix - MLP (MFCC)")
        self.analyze_confusion_matrix(cm, "MLP")
        return model, all_preds, all_labels, accuracy

    def _plot_curves(self, losses, accuracies):
        """Plots training loss and accuracy curves over epochs."""
        epochs = range(1, len(losses) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs, losses, "b-", linewidth=2)
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("CrossEntropy Loss")
        ax1.grid(True)

        ax2.plot(epochs, [a * 100 for a in accuracies], "g-", linewidth=2)
        ax2.set_title("Training Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.grid(True)

        plt.suptitle("MLP Training Curves")
        plt.tight_layout()
        plt.show()

    def run_all(self):
        """
        Full Approach 1 pipeline:
            1. Extract MFCC features
            2. Stratified 70/30 train/test split
            3. Fit StandardScaler on train only (no data leakage)
            4. Train + evaluate SVM
            5. Train + evaluate MLP
            6. Save results to JSON for use in Approach 2
        """
        X, y = self.extract_all_features()
        if X is None:
            return

        # Stratified split: balanced genres in both sets
        # Reference: Tzanetakis & Cook (2002).
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=SEED, stratify=y
        )

        # Fit scaler on training data only — prevents data leakage
        # Reference: Pedregosa et al. (2011). Scikit-learn. JMLR.
        scaler = StandardScaler()
        scaler.fit(X_train)

        # Run SVM
        _, svm_preds, svm_acc = self.run_svm(X_train, X_test, y_train, y_test, scaler)

        # Run MLP
        _, mlp_preds, _, mlp_acc = self.run_mlp(X_train, X_test, y_train, y_test, scaler)

        # Print comparison
        print("\n" + "=" * 50)
        print("       APPROACH 1 — FINAL RESULTS")
        print("=" * 50)
        print(f"  SVM : {svm_acc * 100:.2f}%")
        print(f"  MLP : {mlp_acc * 100:.2f}%")
        print("=" * 50)

        # Save results to JSON so Approach 2 can load them
        # This avoids hardcoding accuracy values
        results = {"svm_accuracy": round(svm_acc * 100, 2), "mlp_accuracy": round(mlp_acc * 100, 2)}
        with open("approach1_results.json", "w") as f:
            json.dump(results, f, indent=4)

        print("\nResults saved to approach1_results.json")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    path = r"D:\MACS\Machin_learning&Big_data\Project\Data\genres_original"
    workflow = Workflow(path)
    workflow.run_all()
