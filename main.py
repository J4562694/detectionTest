from app.model.rtspStream import AiModelPath
from app.config import videoPath
from ultralytics import YOLO
import supervision as sv
import argparse
import cv2

def detectionTest(videoName):
        """
        車牌辨識(待完成...)
        """
        # 初始化Sort追蹤器和YOLO模型
        license_plate_detector = YOLO(AiModelPath.licensePlateDetector)
        txtPath = "./licensePlate.txt"
        tracker = sv.ByteTrack()
        boxAnnotator = sv.BoxAnnotator(
            thickness = 2,
            text_thickness = 2,
            text_scale = 2
        )

        # 定義感興趣的類別ID
        vehicles = [2, 3, 5, 7]
        cam = cv2.VideoCapture(videoPath.videoPath + videoName)

        while True:
            ret, frame = cam.read()
            if not ret:
                break

            # 使用YOLO模型進行車牌偵測
            results = license_plate_detector(frame)
            detection = sv.Detections.from_ultralytics(results[0])
            detection = tracker.update_with_detections(detection)


            labels = [
                f"{license_plate_detector.names[class_id]} {confidence:0.2f}"
                for class_id, confidence in zip(detection.class_id, detection.confidence)
            ]

            frame = boxAnnotator.annotate(scene=frame, detections=detection, labels=labels)

            frame = cv2.resize(frame, (1280, 720))
            cv2.imshow('License Plate Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videoName", help="path to the video file")
    args = parser.parse_args()
    detectionTest(args.videoName)
