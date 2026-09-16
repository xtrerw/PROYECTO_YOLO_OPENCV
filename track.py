import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture("test.mp4")

# 整段视频共用这个集合
seen_ids = set()
min_confidence = 0.7

try:
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    cv2.namedWindow("YOLO", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO", 250, 250)

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 标记是否正在由程序更新进度条
    updating_progress = False
    # 用户拖动进度条时，跳转到指定帧
    def on_progress(position):
        if not updating_progress:
            cap.set(cv2.CAP_PROP_POS_FRAMES, position)

    if total_frames > 1:
        cv2.createTrackbar(
            "Progress",
            "YOLO",
            0,
            total_frames - 1,
            on_progress,
        )



    while True:
        ret, frame = cap.read()

        if not ret:
            print("视频结束或读取中断")
            break

        # 检测车辆并跟踪
        results = model.track(
            frame,
            persist=True,
            tracker="custom_bytetrack.yaml",
            classes=[2, 5, 7],
            conf=min_confidence,
            agnostic_nms=True,
            verbose=False,
        )

        result = results[0]

        # 记录出现过的 ID，同一个 ID 只记录一次
        if result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()

            for track_id in track_ids:
                seen_ids.add(track_id)

        total_count = len(seen_ids)

        # 自动绘制检测框和跟踪 ID
        annotated_frame = result.plot()

        text = f"Total: {total_count}"
        # 黑色描边
        cv2.putText(
            annotated_frame, text, (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5, (0, 0, 0), 6, cv2.LINE_AA,
        )

        # 白色文字
        cv2.putText(
            annotated_frame, text, (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5, (245, 73, 39), 2, cv2.LINE_AA,
        )

        cv2.imshow("YOLO", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        print(f"已处理视频部分的累计车辆 ID 数：{len(seen_ids)}")

finally:
    cap.release()
    cv2.destroyAllWindows()