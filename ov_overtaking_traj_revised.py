import numpy as np

def find_overtaking_indices(overtake_distance_threshold):
    """ obs.npy, actions.npy를 사용하여 추월 인덱스 탐색 후 저장 """

    # 저장된 데이터 불러오기
    obs_data = np.load("obs.npy")
    action_data = np.load("actions.npy")

    # 뒤쪽 차량과의 거리(y축 속도로 가정)
    lane_side_back_distance = obs_data[:, 3]  # y 속도(vy) 값 사용

    # 추월이 발생한 프레임 찾기
    overtake_index_side = np.squeeze(np.argwhere(
        (lane_side_back_distance > overtake_distance_threshold[0]) & 
        (lane_side_back_distance < overtake_distance_threshold[1])
    ))

    # 추월 인덱스 저장
    if overtake_index_side is not None and len(overtake_index_side) > 0:
        np.save("overtaking_indices.npy", overtake_index_side.tolist())
        print(f"✅ 추월 인덱스 저장 완료: {len(overtake_index_side)}개 발견")
    else:
        print("❌ 추월 인덱스 없음")

    return overtake_index_side

# 실행 (추월 감지 기준 설정)
if __name__ == "__main__":
    overtake_distance_threshold = [5, 20]  # 🔥 추월 감지 기준
    find_overtaking_indices(overtake_distance_threshold)