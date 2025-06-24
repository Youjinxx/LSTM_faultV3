## ============================
## 0623, 정상적으로 작동했던 코드. config.py를 수정하면서 혹시 몰라 주석 처리..
## ============================

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from src.model import LSTMClassifier
from src.dataset import LSTMFaultDataset
from torch.utils.data import DataLoader

# ===============================
# 🔧 설정값 (train.py와 반드시 동일하게!)
# ===============================
INPUT_SIZE     = 3
HIDDEN_SIZE    = 32
NUM_LAYERS     = 2
DROPOUT        = 0.3
OUTPUT_SIZE    = 2
BATCH_SIZE     = 32
VAL_RATIO      = 0.15
TEST_RATIO     = 0.15
SEED           = 42

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
    X, y, test_size=VAL_RATIO + TEST_RATIO, stratify=y, random_state=SEED
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
    stratify=y_temp,
    random_state=SEED
)

# ===============================
# 2. Test Dataset & DataLoader
# ===============================
test_dataset = LSTMFaultDataset(X_test, y_test)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===============================
# 3. 모델 선언 및 best_model.pth 불러오기
# ===============================
model = LSTMClassifier(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    output_size=OUTPUT_SIZE,
    dropout=DROPOUT
).to(device)

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# ===============================
# 4. 예측 수행
# ===============================
y_true = []
y_pred = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        output = model(X_batch)
        preds = output.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y_batch.numpy())

# ===============================
# 5. 성능 평가 지표 출력
# ===============================
acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec  = recall_score(y_true, y_pred)
f1   = f1_score(y_true, y_pred)

print(f"\n📊 Test Accuracy : {acc:.4f}")
print(f"📊 Precision     : {prec:.4f}")
print(f"📊 Recall        : {rec:.4f}")
print(f"📊 F1-score      : {f1:.4f}")

# ===============================
# 6. Confusion Matrix 시각화
# ===============================
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fault"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()


# import numpy as np
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# from model import LSTMClassifier
# from dataset import LSTMFaultDataset
# from config import MODEL, TRAIN  # ✅ 설정 import
# from torch.utils.data import DataLoader

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
#     X, y,
#     test_size=TRAIN["val_ratio"] + TRAIN["test_ratio"],
#     stratify=y,
#     random_state=TRAIN["seed"]
# )

# X_val, X_test, y_val, y_test = train_test_split(
#     X_temp, y_temp,
#     test_size=TRAIN["test_ratio"] / (TRAIN["val_ratio"] + TRAIN["test_ratio"]),
#     stratify=y_temp,
#     random_state=TRAIN["seed"]
# )

# # ===============================
# # 2. Test Dataset & DataLoader
# # ===============================
# test_dataset = LSTMFaultDataset(X_test, y_test)
# test_loader  = DataLoader(test_dataset, batch_size=TRAIN["batch_size"], shuffle=False)

# # ===============================
# # 3. 모델 선언 및 best_model.pth 불러오기
# # ===============================
# model = LSTMClassifier(
#     input_size=MODEL["input_size"],
#     hidden_size=MODEL["hidden_size"],
#     num_layers=MODEL["num_layers"],
#     output_size=MODEL["output_size"],
#     dropout=MODEL["dropout"]
# ).to(device)

# model.load_state_dict(torch.load("best_model.pth", map_location=device))
# model.eval()

# # ===============================
# # 4. 예측 수행
# # ===============================
# y_true = []
# y_pred = []

# with torch.no_grad():
#     for X_batch, y_batch in test_loader:
#         X_batch = X_batch.to(device)
#         output = model(X_batch)
#         preds = output.argmax(dim=1).cpu().numpy()
#         y_pred.extend(preds)
#         y_true.extend(y_batch.numpy())

# # ===============================
# # 5. 성능 평가 지표 출력
# # ===============================
# acc  = accuracy_score(y_true, y_pred)
# prec = precision_score(y_true, y_pred)
# rec  = recall_score(y_true, y_pred)
# f1   = f1_score(y_true, y_pred)

# print(f"\n📊 Test Accuracy : {acc:.4f}")
# print(f"📊 Precision     : {prec:.4f}")
# print(f"📊 Recall        : {rec:.4f}")
# print(f"📊 F1-score      : {f1:.4f}")

# # ===============================
# # 6. Confusion Matrix 시각화
# # ===============================
# cm = confusion_matrix(y_true, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fault"])
# disp.plot(cmap=plt.cm.Blues)
# plt.title("Confusion Matrix (Test Set)")
# plt.tight_layout()
# plt.savefig("confusion_matrix.png")
# plt.show()
