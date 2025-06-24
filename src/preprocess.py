import os
import numpy as np
import pandas as pd

# ========================
# 🧩 사용자 설정 파라미터
# ========================

# 사용할 컬럼 순서 (순서 고정)
USE_COLS = ["ia_n", "ib_n", "ic_n", "FF"]

# 샘플링 주파수 (Hz) → 200kHz
FS = 200000  # 고정

# 0초~0.03초 제거 → 0.03초 = 6000 샘플
CUTOFF_START_INDEX = int(0.03 * FS)

# 윈도우 길이 설정 (단위: 샘플 개수)
WIN_SIZE = 1200       # = 6ms

# 윈도우 stride (겹치는 정도 설정 가능)
STRIDE = 1200         # = 6ms마다 한 번 추출 (겹치지 않음)

# Fault 판단 기준: 윈도우 내 FF 평균값 > 이 값이면 Fault(1)
FAULT_THRESHOLD = 0.5


# ========================
# 📂 CSV 로딩 및 슬라이딩 함수
# ========================

def load_csv(filepath):
    """
    CSV 파일을 불러오고, 필요한 열만 numpy array로 반환합니다.
    - 사용 전 0초~0.03초 구간을 잘라냅니다.
    """
    df = pd.read_csv(filepath)

    # 시뮬레이션 초반 0.03초 제거 (6000개 샘플)
    df = df.iloc[CUTOFF_START_INDEX:]

    # 필요한 열만 선택 (ia_n, ib_n, ic_n, FF)
    data = df[USE_COLS].values  # shape: (샘플 수, 4)
    return data


def create_windows(data, win_size=WIN_SIZE, stride=STRIDE):
    """
    슬라이딩 윈도우를 적용하여 LSTM 입력 시퀀스를 생성합니다.
    """
    X, y = [], []  # 입력값 X, 라벨 y
    for i in range(0, len(data) - win_size, stride):
        window = data[i:i+win_size]
        x_window = window[:, :3]            # ia_n, ib_n, ic_n
        y_window = window[:, 3]             # FF
        y_label = 1 if y_window.mean() > FAULT_THRESHOLD else 0
        X.append(x_window)
        y.append(y_label)
    return np.array(X), np.array(y)


def load_all_data(base_dir):
    """
    A/B/C 상 폴더 내 모든 CSV 데이터를 불러오고,
    LSTM 학습용 (X, y) 윈도우 배열로 반환합니다.
    """
    all_X, all_y = [], []
    for phase_folder in sorted(os.listdir(base_dir)):
        phase_path = os.path.join(base_dir, phase_folder)
        if not os.path.isdir(phase_path):
            continue
        for fname in sorted(os.listdir(phase_path)):
            fpath = os.path.join(phase_path, fname)
            if not fname.endswith(".csv"):
                continue
            try:
                raw = load_csv(fpath)
                X, y = create_windows(raw)
                all_X.append(X)
                all_y.append(y)
            except Exception as e:
                print(f"[Error loading] {fpath}: {e}")
    # shape: (총 윈도우 수, WIN_SIZE, 3), (총 윈도우 수,)
    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)


# ========================
# 🚀 테스트 실행 및 정규화
# ========================
if __name__ == "__main__":
    X, y = load_all_data("./data")

    # 정규화
    from sklearn.preprocessing import MinMaxScaler
    import joblib

    X_reshaped = X.reshape(-1, X.shape[-1])
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X_reshaped)
    X = X_scaled.reshape(X.shape)

    # 저장
    joblib.dump(scaler, "scaler.pkl")
    np.save("X.npy", X)
    np.save("y.npy", y)

    print(f"✔ 전체 윈도우 수: {len(X)}")
    print(f"✔ X shape: {X.shape}")
    print(f"✔ y shape: {y.shape}")
    print(f"✔ 클래스 분포 (0=정상, 1=Fault): {np.bincount(y)}")

