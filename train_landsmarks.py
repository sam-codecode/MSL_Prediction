import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import HandSignModel

# --- Load Dataset ---
df = pd.read_csv("data/all_labels.csv")
print(f"✅ Dataset shape: {df.shape}")

X = df.drop(columns=["label", "handedness"]).values  # drop non-numeric
y = df["label"].astype("category").cat.codes.values
label_mapping = dict(enumerate(df["label"].astype("category").cat.categories))
joblib.dump(label_mapping, "label_mapping.pkl")
print(f"✅ Label mapping saved: {label_mapping}")

# --- Normalize features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")
print("✅ Features normalized & scaler saved")

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(X_test_tensor, y_test_tensor), batch_size=64, shuffle=False
)

# --- Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HandSignModel(input_size=X_train.shape[1], hidden_size=256, num_classes=len(label_mapping)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training Loop ---
epochs = 50
for epoch in range(1, epochs + 1):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    # --- Test ---
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total

    print(f"Epoch [{epoch}/{epochs}], Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

# --- Save model ---
torch.save(model.state_dict(), "hand_sign_model.pth")
print("✅ Model saved as hand_sign_model.pth")
