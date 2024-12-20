import time
from pynput.mouse import Listener, Button
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from threading import Event

# 마우스 좌표 저장
positions = []
stop_event = Event()  # 종료 이벤트
tracking = False

#def on_move(x, y):
#    """마우스 움직임 분석"""
#    positions.append((x, y))

def on_click(x, y, button, pressed):
    """마우스 클릭 분석"""
    global tracking
    if button == Button.left:
        if pressed:
            tracking = True
        else:
            tracking = False

def on_move(x, y):
    """마우스 움직임 분석"""
    global tracking
    if tracking:
        positions.append((x, y))

def main():
    """마우스 움직임 추적 및 히트맵 생성"""
    print("마우스 추적 시작. 오른쪽 클릭을 통해 이동하세요. 종료하려면 Ctrl+C를 누르세요.")
    listener = Listener(on_move=on_move, on_click=on_click)
    listener.start()  # 백그라운드에서 Listener 실행

    try:
        while not stop_event.is_set():  # 종료 이벤트를 기다림
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n마우스 추적 종료. 히트맵 생성 중...")
        stop_event.set()
        listener.stop()

        # 데이터 정리
        if not positions:
            print("수집된 데이터가 없습니다.")
            return

        x_coords, y_coords = zip(*positions)

        # 화면 크기 확인 (데이터에 따라 수동 조정 가능)
        screen_width, screen_height = max(x_coords), max(y_coords)

        # 히트맵 데이터 준비
        heatmap, xedges, yedges = np.histogram2d(
            x_coords, y_coords, bins=(100, 100),
            range=[[0, screen_width], [0, screen_height]]
        )

        # 히트맵 시각화
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            heatmap.T,  # 히트맵 데이터
            cmap="Reds",  # 색상 맵을 더 부드럽게 설정
            cbar=True,
            xticklabels=False,
            yticklabels=False,
            linewidths=0.1,  # 각 셀 사이의 경계선 두께
            alpha=0.7,  # 히트맵의 투명도 조정
            square=False,  # 정사각형으로 표시하지 않음
            linecolor='black'  # 경계선 색상 설정
        )
        plt.title("Mouse Movement Heatmap")
        plt.xlabel("Screen X")
        plt.ylabel("Screen Y")
        plt.show()

if __name__ == "__main__":
    main()
