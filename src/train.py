import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from src.preprocess import preprocess_dataset
from src.model import LSTMModel
from src.train_utils import train_model
import numpy as np
import os
os.makedirs("./models", exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
print("Device:", device)

import random

# Set random seed for PyTorch
seed = 42
torch.manual_seed(seed)  # CPU
torch.cuda.manual_seed(seed)  # GPU (for current device)
torch.cuda.manual_seed_all(seed)  # GPU (for all devices)

# Set random seed for Python's random module
random.seed(seed)

# Set random seed for NumPy
np.random.seed(seed)

# Ensure deterministic behavior (optional)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load data

X, y = preprocess_dataset(
    "./datasets/cdGII_seqs.txt",
    "./datasets/cdGII_labels.txt"
)

# Generate shuffled indices
idx = np.random.permutation(X.shape[0])

# Compute mean and standard deviation
mean_y = np.mean(np.array(y), axis=0)
std_y = np.std(np.array(y), axis=0)

# Standardize y
y_standardized = (y - mean_y) / std_y

# Apply the shuffled indices
X_shuffled = X[idx]
y_shuffled = y_standardized[idx]


train_size = int(0.85 * len(X_shuffled))
X_train, X_test = X_shuffled[:train_size], X_shuffled[train_size:]
y_train, y_test = y_shuffled[:train_size], y_shuffled[train_size:]

# convert to pytorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_valid_tensor = torch.tensor(X_test, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# save train/validation split
torch.save(
    {
        "X_train": X_train_tensor,
        "y_train": y_train_tensor,
        "X_valid": X_valid_tensor,
        "y_valid": y_valid_tensor,
    },
    "./data/processed/data_split.pt",
)

# create dataloader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=False)


# model
model = LSTMModel(
    input_size=113,
    hidden_size=512,
    output_size=1
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4
)

train_model(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    epochs=100,
    patience=20,
    save_path="./models/best_model.pt"
)
