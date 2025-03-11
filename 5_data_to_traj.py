import numpy as np
import pandas as pd

def process_csv_to_npy(csv_path):
    """ 특정 CSV 파일을 불러와 numpy 배열로 변환 후 저장 """
    
    # CSV 데이터 불러오기
    df = pd.read_csv(csv_path)

    # 필요한 컬럼 선택 (사용자 CSV 파일 구조에 맞게 조정)
    obs_data = df[['x', 'y', 'vx', 'vy']].to_numpy()  # 예: 위치(x, y), 속도(vx, vy)
    action_data = df['control'].to_numpy()  # 예: 차량의 조향/가속/제동 명령

    # 데이터 저장
    np.save("obs.npy", obs_data)
    np.save("actions.npy", action_data)

    print(f"✅ 데이터 변환 완료: {csv_path}")
    print(f"🔹 obs.npy 저장: {obs_data.shape}, actions.npy 저장: {action_data.shape}")

# 실행 (사용자 CSV 경로 입력)
if __name__ == "__main__":
    csv_path = "highway_task/data/345/01_episode0.csv"  # 🔥 CSV 파일 경로 변경 가능
    process_csv_to_npy(csv_path)