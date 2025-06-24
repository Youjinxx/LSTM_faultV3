import torch
from torch.utils.data import Dataset
import numpy as np

# ===============================
# 🧠 LSTM Fault Detection Dataset
# ===============================

# 어딘가에 Class를 import해서 사용하는 용도임, 단독으로 터미널에서 실행할 필요는 없음. 

class LSTMFaultDataset(Dataset):
    """
    전처리된 시퀀스 데이터를 PyTorch LSTM 학습용으로 제공하는 Dataset 클래스입니다.
    데이터들은 윈도우 단위로 라벨링 및 정규화되었음.
    """

    def __init__(self, X, y, dtype=torch.float32):
        """
        Dataset 초기화 함수

        ✔ X: numpy array, shape = (윈도우 수, 시퀀스 길이, feature 수)
        ✔ y: numpy array, shape = (윈도우 수,)
        ✔ dtype: Tensor 변환 시 사용할 데이터 타입 (기본: torch.float32)

        🧩 변경 가능한 파라미터:
        - dtype: float16 / float32 / float64 등 지정 가능
        """
        assert X.shape[0] == y.shape[0], "X와 y의 윈도우 수가 다릅니다!"

        self.X = torch.tensor(X, dtype=dtype)     # (윈도우 수, 시퀀스 길이, 3)
        self.y = torch.tensor(y, dtype=torch.long)  # 라벨은 정수형 (0 or 1)

    def __len__(self):
        """
        전체 샘플 수 반환
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        주어진 인덱스(idx)에 해당하는 X, y를 반환
        - X: (시퀀스 길이, feature 수)  → (1200, 3)
        - y: 정수 라벨 (0 or 1)
        """
        return self.X[idx], self.y[idx]
