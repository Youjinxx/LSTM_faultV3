import torch
import torch.nn as nn
# from config import MODEL

# ============================
# 🧠 LSTM 기반 Fault 분류 모델
# ============================

# ============================
# 0623, 정상적으로 작동했던 코드. config.py를 수정하면서 혹시 몰라 주석 처리..
# ============================
class LSTMClassifier(nn.Module):
    def __init__(self,
                 input_size=3,         #  입력 차원: ia_n, ib_n, ic_n (기본 = 3)
                 hidden_size=32,       #  LSTM 셀의 hidden 상태 크기
                 num_layers=2,         #  LSTM 레이어 개수
                 output_size=2,        #  출력 클래스 수 (정상/고장 = 2)
                 dropout=0.3):         #  Dropout 비율 (층 사이에 적용)

        super(LSTMClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Layer 구성
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,  # 1층일 때는 dropout X
            batch_first=True  # (batch, seq_len, input_size) 형태 입력
        )

        # 마지막 시점의 hidden state만 받아서 분류기로 연결
        self.fc = nn.Linear(hidden_size, output_size)  # (batch, output_size)

    def forward(self, x):
        """
        x: shape (batch_size, seq_len, input_size)
        output: shape (batch_size, output_size)
        """
        # LSTM 통과
        lstm_out, (hn, cn) = self.lstm(x)  # lstm_out: (batch, seq_len, hidden)

        # 마지막 시점의 출력만 사용 (lstm_out[:, -1, :])
        final_hidden = lstm_out[:, -1, :]  # shape: (batch, hidden)

        # FC 레이어 통과 (분류기)
        out = self.fc(final_hidden)       # shape: (batch, output_size)

        return out


# # ============================
# # 🧠 LSTM 기반 Fault 분류 모델
# # ============================

# class LSTMClassifier(nn.Module):
#     def __init__(self,
#                  input_size=MODEL["input_size"],
#                  hidden_size=MODEL["hidden_size"],
#                  num_layers=MODEL["num_layers"],
#                  output_size=MODEL["output_size"],
#                  dropout=MODEL["dropout"]):

#         super(LSTMClassifier, self).__init__()

#         self.hidden_size = hidden_size
#         self.num_layers = num_layers

#         # LSTM Layer 구성
#         self.lstm = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             dropout=dropout if num_layers > 1 else 0.0,
#             batch_first=True  # (batch, seq_len, input_size)
#         )

#         # 마지막 시점의 hidden state만 받아서 분류기로 연결
#         self.fc = nn.Linear(hidden_size, output_size)  # (batch, output_size)

#     def forward(self, x):
#         """
#         x: shape (batch_size, seq_len, input_size)
#         output: shape (batch_size, output_size)
#         """
#         # LSTM 통과
#         lstm_out, _ = self.lstm(x)  # lstm_out: (batch, seq_len, hidden)

#         # 마지막 시점의 출력만 사용
#         final_hidden = lstm_out[:, -1, :]  # (batch, hidden)

#         # FC 레이어 통과
#         out = self.fc(final_hidden)       # (batch, output_size)

#         return out
