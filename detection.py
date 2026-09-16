import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture("test.mp4")

# 想显示哪些类别，就写在这里
min_confidence = 0.7

try:
    if not cap.isOpened():
        raise  RuntimeError("Cannot open video")

    # 创建可调整大小的窗口，只需要执行一次
    cv2.namedWindow("YOLO", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO", 640, 360)

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
        ret,frame = cap.read()

        if not ret:
            break

        # 读取后的位置通常指向下一帧，所以减 1
        current_frame = max(
            0,
            int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1,
        )
        # 使用同一个置信度阈值进行推理
        results = model(frame,
                        classes=[2,5,7],
                        conf=min_confidence,
                        #去除高度重叠的重复框时，不区分类别
                        agnostic_nms=True,
                        verbose=False)
        result = results[0]
        # 提取每个目标的信息
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            name = result.names[class_id]
            confidence = float(box.conf[0].item())
            coordinates = box.xyxy[0].tolist()

            print(name, confidence, coordinates)

        # 当前帧检测到的人数
        count = len(result.boxes)
        print(f"当前车辆数：{count}")

        # 自动绘制检测框、类别和置信度
        annotated_frame = result.plot()
        cv2.imshow("YOLO", annotated_frame)

        if total_frames > 1:
            updating_progress = True
            try:
                cv2.setTrackbarPos("Progress", "YOLO", current_frame)
            finally:
                updating_progress = False
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

