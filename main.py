import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture("test.mp4")
try:
    if not cap.isOpened():
        raise  RuntimeError("Cannot open webcam")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("The video is finished, can´t read it")
            break

        results = model(frame)

        frame = results[0].plot()

        cv2.imshow("YOLO", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            name = result.names[cls]

            print(name)
finally:
    cap.release()
    cv2.destroyAllWindows()