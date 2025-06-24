# ============================
# 0623, 정상적으로 작동했던 코드. config.py를 수정하면서 혹시 몰라 주석 처리..
# ============================

 # ===============================
# # 🧩 사용자 설정 가능한 파라미터
# # ===============================

# import numpy as np
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
# from sklearn.model_selection import train_test_split

# from src.dataset import LSTMFaultDataset
# from src.model import LSTMClassifier

# # ===============================
# # 🧩 사용자 설정 가능한 파라미터
# # ===============================

# BATCH_SIZE     = 32         # 한 번에 처리할 데이터 수, 일반적으로 2의 제곱근으로 처리
# NUM_EPOCHS     = 100000       # 전체 학습 반복 횟수
# LEARNING_RATE  = 1e-2       # 모델 학습 속도

# # ⚠️ model.py 와 반드시 일치/ 0623: train.py에서 끌어쓰도록 수정하였음.// 순환회귀.. 
# HIDDEN_SIZE    = 32
# NUM_LAYERS     = 2
# DROPOUT        = 0.3
# INPUT_SIZE     = 3
# OUTPUT_SIZE    = 2

# VAL_RATIO      = 0.15
# TEST_RATIO     = 0.15
# SEED           = 42

# # ✅ 조기 종료 조건
# LOSS_THRESHOLD = 0.005    # val_loss가 이 값보다 작으면
# PATIENCE       = 3       # 연속 PATIENCE번 만족 시 조기 종료

# # ===============================
# # ⚙️ 디바이스 설정
# # ===============================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ===============================
# # 1. 데이터 불러오기 및 분할
# # ===============================
# X = np.load("X.npy")
# y = np.load("y.npy")

# X_train, X_temp, y_train, y_temp = train_test_split(
#     X, 
#     y, 
#     test_size=VAL_RATIO + TEST_RATIO, 
#     stratify=y, 
#     random_state=SEED
# )

# X_val, X_test, y_val, y_test = train_test_split(
#     X_temp, 
#     y_temp,
#     test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
#     stratify=y_temp,
#     random_state=SEED
# )

# print(f"✔ Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# # ===============================
# # 2. Dataset & DataLoader 구성
# # ===============================
# train_dataset = LSTMFaultDataset(X_train, y_train)
# val_dataset   = LSTMFaultDataset(X_val, y_val)
# test_dataset  = LSTMFaultDataset(X_test, y_test)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
# test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# # ===============================
# # 3. 모델, 손실함수, 옵티마이저 구성
# # ===============================
# model = LSTMClassifier(
#     input_size=INPUT_SIZE,
#     hidden_size=HIDDEN_SIZE,
#     num_layers=NUM_LAYERS,
#     output_size=OUTPUT_SIZE,
#     dropout=DROPOUT
# ).to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# # ===============================
# # 4. 학습 루프 + 조기 종료
# # ===============================
# best_val_acc = 0.0
# early_stop_counter = 0

# train_losses = []
# val_losses = []


# for epoch in range(NUM_EPOCHS):
#     model.train()
#     train_loss, correct = 0.0, 0

#     for X_batch, y_batch in train_loader:
#         X_batch = X_batch.to(device)
#         y_batch = y_batch.to(device)

#         output = model(X_batch)
#         loss = criterion(output, y_batch)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item() * X_batch.size(0)
#         pred = output.argmax(dim=1)
#         correct += (pred == y_batch).sum().item()

#     avg_train_loss = train_loss / len(train_loader.dataset)
#     train_acc = correct / len(train_loader.dataset)


#     # ===== Validation =====
#     model.eval()
#     val_loss, val_correct = 0.0, 0

#     with torch.no_grad():
#         for X_batch, y_batch in val_loader:
#             X_batch = X_batch.to(device)
#             y_batch = y_batch.to(device)

#             output = model(X_batch)
#             loss = criterion(output, y_batch)

#             val_loss += loss.item() * X_batch.size(0)
#             pred = output.argmax(dim=1)
#             val_correct += (pred == y_batch).sum().item()

#     avg_val_loss = val_loss / len(val_loader.dataset)
#     val_acc = val_correct / len(val_loader.dataset)

#     train_losses.append(avg_train_loss)
#     val_losses.append(avg_val_loss)

#     print(f"📘 Epoch [{epoch+1}/{NUM_EPOCHS}] "
#           f"Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} || "
#           f"Val Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f}")

#     # 💾 Best 모델 저장
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         torch.save(model.state_dict(), "best_model.pth")
#         print("✅ Best model saved.")

#     # 🛑 조기 종료 조건 확인
#     if avg_val_loss < LOSS_THRESHOLD:
#         early_stop_counter += 1
#         print(f"⚠️ val_loss < {LOSS_THRESHOLD:.4f} 만족 ({early_stop_counter}/{PATIENCE})")
#         if early_stop_counter >= PATIENCE:
#             print(f"🛑 Early stopping triggered at epoch {epoch+1}")
#             break
#     else:
#         early_stop_counter = 0

# print("🎉 학습 완료!")

# # ===============================
# # 5. 학습 곡선 시각화
# # ===============================
# plt.figure(figsize=(8, 5))
# plt.plot(train_losses, label="Train Loss", color='blue')
# plt.plot(val_losses, label="Val Loss", color='orange')
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training / Validation Loss Curve")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("loss_curve.png")  # 이미지 저장
# plt.show()

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import LSTMFaultDataset
from model import LSTMClassifier
from config import MODEL, TRAIN  # ✅ 파라미터 통합 import

# ===============================
# ⚙️ 디바이스 설정
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# 1. 데이터 불러오기 및 분할
# ===============================
X = np.load("X.npy")
y = np.load("y.npy")

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=TRAIN["val_ratio"] + TRAIN["test_ratio"],
    stratify=y,
    random_state=TRAIN["seed"]
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=TRAIN["test_ratio"] / (TRAIN["val_ratio"] + TRAIN["test_ratio"]),
    stratify=y_temp,
    random_state=TRAIN["seed"]
)

print(f"✔ Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ===============================
# 2. Dataset & DataLoader 구성
# ===============================
train_dataset = LSTMFaultDataset(X_train, y_train)
val_dataset   = LSTMFaultDataset(X_val, y_val)
test_dataset  = LSTMFaultDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=TRAIN["batch_size"], shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=TRAIN["batch_size"], shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=TRAIN["batch_size"], shuffle=False)

# ===============================
# 3. 모델, 손실함수, 옵티마이저 구성
# ===============================
model = LSTMClassifier(
    input_size=MODEL["input_size"],
    hidden_size=MODEL["hidden_size"],
    num_layers=MODEL["num_layers"],
    output_size=MODEL["output_size"],
    dropout=MODEL["dropout"]
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN["learning_rate"])

# ===============================
# 4. 학습 루프 + 조기 종료
# ===============================
best_val_acc = 0.0
early_stop_counter = 0
train_losses = []
val_losses = []

for epoch in range(TRAIN["num_epochs"]):
    model.train()
    train_loss, correct = 0.0, 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        output = model(X_batch)
        loss = criterion(output, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X_batch.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == y_batch).sum().item()

    avg_train_loss = train_loss / len(train_loader.dataset)
    train_acc = correct / len(train_loader.dataset)

    # ===== Validation =====
    model.eval()
    val_loss, val_correct = 0.0, 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            output = model(X_batch)
            loss = criterion(output, y_batch)

            val_loss += loss.item() * X_batch.size(0)
            pred = output.argmax(dim=1)
            val_correct += (pred == y_batch).sum().item()

    avg_val_loss = val_loss / len(val_loader.dataset)
    val_acc = val_correct / len(val_loader.dataset)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    print(f"📘 Epoch [{epoch+1}/{TRAIN['num_epochs']}] "
          f"Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} || "
          f"Val Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f}")

    # 💾 Best 모델 저장
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Best model saved.")

    # 🛑 조기 종료 조건 확인
    if avg_val_loss < TRAIN["loss_threshold"]:
        early_stop_counter += 1
        print(f"⚠️ val_loss < {TRAIN['loss_threshold']:.4f} 만족 ({early_stop_counter}/{TRAIN['patience']})")
        if early_stop_counter >= TRAIN["patience"]:
            print(f"🛑 Early stopping triggered at epoch {epoch+1}")
            break
    else:
        early_stop_counter = 0

print("🎉 학습 완료!")

# ===============================
# 5. 학습 곡선 시각화
# ===============================
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss", color='blue')
plt.plot(val_losses, label="Val Loss", color='orange')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training / Validation Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()
