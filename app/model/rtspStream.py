from ultralytics import YOLO
from supervision.detection.line_counter import LineZone, LineZoneAnnotator
from app.config import AiModelPath
from app.config import downloadVideoPath
import supervision as sv
import numpy as np
import time
import cv2
import threading
import pytesseract

class Camera:
    def __init__(self, streamUrl: str):
        self.camera = cv2.VideoCapture(streamUrl)
        self.lock = threading.Lock()

    def getFrame(self):
        success, frame = self.camera.read()
        if not success:
            return None
        ret, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()

    def release(self):
        self.camera.release()

    def genFrames(self):
        while True:
            with self.lock:
                frame = self.getFrame()
            if frame is None:
                break
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
    def boxModelDetect(self, modelPath: str, classList: list=[], thickness: int=2, textThickness: int=2, textScale: int=1, confidence: float=0):
        """
        ### 預測框
        - modelPath: AI模組路徑(.pt)
        - thickness: 預測框線粗度
        - textThickness: 文字粗度
        - textScale: 文字大小
        - confidence: 模型偵測的閥值
        """
        boxAnnotator = sv.BoxAnnotator(
            thickness = thickness,
            text_thickness = textThickness,
            text_scale = textScale
        )
        model = YOLO(modelPath)
        tracker = sv.ByteTrack()
        while True:
            ret, frame = self.camera.read()

            if not ret:
                break

            if classList == []:
                results = model(frame)[0]
            else:
                results = model(frame, classes=classList, verbose=False)[0]

            detection = sv.Detections.from_ultralytics(results)

            # 設定信心度
            if confidence > 0:
                detection = detection[detection.confidence > 0]
            detection = tracker.update_with_detections(detection)

            labels = [
                f"{model.names[class_id]} {confidence:0.2f}"
                for class_id, confidence in zip(detection.class_id, detection.confidence)
            ]

            frame = boxAnnotator.annotate(scene=frame, detections=detection, labels=labels)
            
            # 將畫面轉成jpg格式
            ret, buffer = cv2.imencode('.jpg', frame)
            jpgFrame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpgFrame + b'\r\n')
    
    def lineModelDetect(self, modelPath: str, classList: list=[], lineStartX: int=0, lineStartY: int=0, lineEndX: int=0, lineEndY: int=0, confidence: float=0):
        """
            modelPath: AI模組路徑(.pt)
            classList: 偵測項目
            lineStartX: 偵測線起始點X軸
            lineStartY: 偵測線起始點Y軸
            lineEndX: 偵測線結束點X軸
            lineEndY: 偵測線結束點Y軸
            confidence: 偵測閥值(模型偵測高過於閥值才會顯示偵測)
        """
        model = YOLO(modelPath)
        # 線的基礎設定
        lineStart = sv.Point(lineStartX, lineStartY)
        lineEnd = sv.Point(lineEndX, lineEndY)
        lineCounter = LineZone(start=lineStart, end=lineEnd)
        LineAnnotator = LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=0.5)
        tracker = sv.ByteTrack()

        # 偵測框設定
        boxAnnotator = sv.BoxAnnotator(
            thickness = 2,
            text_thickness = 2,
            text_scale = 1
        )

        while True:
            
            ret, frame = self.camera.read()

            if ret:
                if classList == []:
                    results = model(frame)[0]
                else:
                    results = model(frame, classes=classList, verbose=False)[0]
                detection = sv.Detections.from_ultralytics(results)
                # 設定信心度
                if confidence > 0:
                    detection = detection[detection.confidence > confidence]
                detection = tracker.update_with_detections(detection)

                labels = [
                    f"{model.names[class_id]} {confidence:0.2f}"
                    for class_id, confidence in zip(detection.class_id, detection.confidence)
                ]
                frame = boxAnnotator.annotate(scene=frame, detections=detection, labels=labels)

                LineAnnotator.annotate(frame=frame, line_counter=lineCounter)
                lineCounter.trigger(detections=detection)
                # print(f"進來人數:{lineCounter.in_count} \n 出去人數:{lineCounter.out_count}")

                # 將畫面轉成jpg格式
                ret, buffer = cv2.imencode('.jpg', frame)
                jpgFrame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpgFrame + b'\r\n')
                
    def countObjectsInRange(self, modelPath: str, classList: list=[], confidence: float=0):
        """
            modelPath: AI模組路徑(.pt)
            classList: 偵測項目
            lineStartX: 偵測線起始點X軸
            lineStartY: 偵測線起始點Y軸
            lineEndX: 偵測線結束點X軸
            lineEndY: 偵測線結束點Y軸
            confidence: 偵測閥值(模型偵測高過於閥值才會顯示偵測)
        """
        model = YOLO(modelPath)

        #畫面偵測範圍的設定
        ret, frame = self.camera.read()
        frameWidth, frameHeight = frame.shape[:2]
        polygon = np.array([
            [351, 428], 
            [614, 835], 
            [1243, 461], 
            [829, 291], 
            [351, 424]
        ])
        Polygon = sv.PolygonZone(polygon=polygon, frame_resolution_wh=[frameWidth, frameHeight])
        PolygonAnnotator = sv.PolygonZoneAnnotator(zone=Polygon, color=sv.Color.white(), thickness=2, text_thickness=2, text_scale=1)

        # 偵測框的設定
        boxAnnotator = sv.BoxAnnotator(
            thickness = 2,
            text_thickness = 2,
            text_scale = 1
        )

        while True:

            ret, frame = self.camera.read()
            
            if classList == []:
                results = model(frame)[0]
            else:
                results = model(frame, classes=classList, verbose=False)[0]
            detection = sv.Detections.from_ultralytics(results)
            # 設定信心度
            if confidence > 0:
                detection = detection[detection.confidence > confidence]
            labels = [
                f"{model.names[class_id]} {confidence:0.2f}"
                for class_id, confidence in zip(detection.class_id, detection.confidence)
            ]
            frame = boxAnnotator.annotate(scene=frame, detections=detection, labels=labels)
            PolygonAnnotator.annotate(scene=frame)
            Polygon.trigger(detections=detection)

            # 將畫面轉成jpg格式
            ret, buffer = cv2.imencode('.jpg', frame)
            jpgFrame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpgFrame + b'\r\n')

    def licensePlateRecognition(self):
        """
        車牌辨識(待完成...)
        """

        # 初始化Sort追蹤器和YOLO模型
        coco_model = YOLO(AiModelPath.yolov8n)
        license_plate_detector = YOLO(AiModelPath.licensePlateDetector)
        txtPath = "./licensePlate.txt"

        # 定義感興趣的類別ID
        vehicles = [2, 3, 5, 7]

        while True:

            ret, frame = self.camera.read()
            # 偵測汽車
            detections = coco_model(frame, classes=vehicles, device='cpu')[0]
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if score < 0.5:
                    continue
                
                # 繪製汽車邊界框
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 5)
                
                # 裁切汽車區域並偵測車牌
                car_region = frame[int(y1):int(y2), int(x1):int(x2)]
                plate_detections = license_plate_detector(car_region, device='cpu')[0]
                for plate_det in plate_detections.boxes.data.tolist():
                    px1, py1, px2, py2, pscore, pclass_id = plate_det
                    if pscore < 0.6:
                        continue
                    
                    # 繪製車牌邊界框（相對於原始影像）
                    cv2.rectangle(frame, (int(x1+px1), int(y1+py1)), (int(x1+px2), int(y1+py2)), (255, 0, 0), 3)
                    
                    absolute_px1 = int(x1 + px1)
                    absolute_py1 = int(y1 + py1)
                    absolute_px2 = int(x1 + px2)
                    absolute_py2 = int(y1 + py2)

                    # 裁切車牌圖像
                    license_plate_crop = frame[absolute_py1:absolute_py2, absolute_px1:absolute_px2]

                    # process license plate
                    # blur = cv2.medianBlur(license_plate_crop, 5)
                    # cv2.imshow('image', blur)
                    # cv2.waitKey(0)

                    # 車牌照片預處理
                    licensePlate = cv2.convertScaleAbs(license_plate_crop, alpha=1.7, beta=0)
                    license_plate_crop_gray = cv2.cvtColor(licensePlate, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 127, 255, cv2.THRESH_BINARY_INV)
                    license_plate_resized = cv2.resize(license_plate_crop_thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

                    # 幫預處理完的照片分割大小(把外框移除幫助偵測)
                    x = 20
                    y = 15
                    w = 275
                    h = 400
                    newResizeImg = license_plate_resized[y:y+h, x:x+w]

                    # pytesseract偵測車牌照片
                    results = pytesseract.image_to_string(newResizeImg, lang='eng', config='--oem 3 --psm 7')
                    license_plate_text = ''

                    if results:
                        with open(txtPath, 'a') as f:
                            f.write(results)

                    text_size = cv2.getTextSize(license_plate_text, cv2.FONT_HERSHEY_SIMPLEX, 5, 2)[0]
                    text_width, text_height = text_size[0], text_size[1]
                    text_x1 = x1
                    text_y1 = y1 - text_height - 10  # 略微向上移動，避免與車牌重疊
                    text_x2 = x1 + text_width
                    text_y2 = y1 - 5  # 與文字間距保持一致性

                    # 畫出文字背景框
                    cv2.rectangle(frame, (int(text_x1), int(text_y1)), (int(text_x2), int(text_y2)), (0, 255, 0), cv2.FILLED)

                    # 將車牌號碼繪製到背景框上
                    cv2.putText(frame, license_plate_text, (int(text_x1), int(y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # 將畫面轉成jpg格式
            ret, buffer = cv2.imencode('.jpg', frame)
            jpgFrame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpgFrame + b'\r\n')
     
    def videoPredict(self, modelPath: str, classList: list=[], thickness: int=2, textThickness: int=2, textScale: int=1, confidence: float=0, filePath: str=""):
        """
        ### 預測框
        - modelPath: AI模組路徑(.pt)
        - thickness: 預測框線粗度
        - textThickness: 文字粗度
        - textScale: 文字大小
        - confidence: 模型偵測的閥值
        """

         # 打开视频文件
        cap = cv2.VideoCapture(filePath)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 定义视频编码器和创建 VideoWriter 对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 根据需要调整编码器
        out = cv2.VideoWriter(f"{downloadVideoPath.videoPath}\\detect.mp4", fourcc, 15.0, (frame_width, frame_height))
        boxAnnotator = sv.BoxAnnotator(
            thickness = thickness,
            text_thickness = textThickness,
            text_scale = textScale
        )
        model = YOLO(modelPath)
        tracker = sv.ByteTrack()
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if classList == []:
                results = model(frame)[0]
            else:
                results = model(frame, classes=classList, verbose=False)[0]

            detection = sv.Detections.from_ultralytics(results)
            # 設定信心度
            if confidence > 0:
                detection = detection[detection.confidence > 0]
            detection = tracker.update_with_detections(detection)

            labels = [
                f"{model.names[class_id]} {confidence:0.2f}"
                for class_id, confidence in zip(detection.class_id, detection.confidence)
            ]

            frame = boxAnnotator.annotate(scene=frame, detections=detection, labels=labels)
            out.write(frame)

        cap.release()
        out.release()
        self.camera.release()

        return True
    
    def detectionCoordinates(self, modelPath: str, classList: list=[], thickness: int=2, textThickness: int=2, textScale: int=1, confidence: float=0):
        boxAnnotator = sv.BoxAnnotator(
                    thickness = thickness,
                    text_thickness = textThickness,
                    text_scale = textScale
                )
        model = YOLO(modelPath)
        tracker = sv.ByteTrack()
        while True:
            ret, frame = self.camera.read()

            if not ret:
                break

            if classList == []:
                results = model(frame)[0]
            else:
                results = model(frame, classes=classList, verbose=False)[0]

            detection = sv.Detections.from_ultralytics(results)

            # 設定信心度
            if confidence > 0:
                detection = detection[detection.confidence > 0]
            detection = tracker.update_with_detections(detection)

            labels = [
                f"{model.names[class_id]} {confidence:0.2f}"
                for class_id, confidence in zip(detection.class_id, detection.confidence)
            ]

            frame = boxAnnotator.annotate(scene=frame, detections=detection, labels=labels)
            
            # 將畫面轉成jpg格式
            ret, buffer = cv2.imencode('.jpg', frame)
            jpgFrame = buffer.tobytes()

    def drawDetectionBoxes(self, detectionData):
        """ 待完成
        ### 回放事件處理function
            predictTime: 事件發生時間點
            x1: 左上角x軸
            x2: 右下角x軸
            y1: 左上角y軸
            y2: 右下角y軸
        """
    
        fpsCounter = 1
        detectionDataIndex = 0

        while True:
            ret, frame = self.camera.read()
            if not ret:
                break

            if fpsCounter == 15:
                fpsCounter = 1
                detectionDataIndex += 1

            try:
                for item in detectionData[detectionDataIndex]:
                    # 畫出偵測框
                    boundingBox = item['boundingBox'].split(',')
                    x1 = int(float(boundingBox[0]))
                    y1 = int(float(boundingBox[1]))
                    x2 = int(float(boundingBox[2]))
                    y2 = int(float(boundingBox[3]))
                    confidence = int(float(item['confidence']) * 100)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 3)
                    
                    # 畫出偵測框的label text
                    (textWidth, textHeight), _ = cv2.getTextSize(f"{item['label']} {item['confidence']}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    padding = 5

                    backgroundWidth = textWidth + padding * 2
                    # backgroundHeight = textHeight + padding * 2

                    bgTopLeft = (x1, y1 - textHeight - padding)
                    bgBottomRight = (x1 + backgroundWidth, y1)

                    cv2.rectangle(frame, bgTopLeft, bgBottomRight, (0, 0, 255), -1)  # -1 代表填充
                    cv2.putText(frame, f"{item['label']} {str(confidence)}%", (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            except IndexError:
                break
            
            fpsCounter += 1
            ret, buffer = cv2.imencode('.jpg', frame)
            jpgFrame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpgFrame + b'\r\n')
