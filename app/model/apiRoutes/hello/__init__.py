from flask import Blueprint, jsonify, Response
from app.module.rtspStream import Camera
from ultralytics import YOLO
from supervision.detection.line_counter import LineZone, LineZoneAnnotator
from app.config import AiModelPath

router = Blueprint('helloRouter', __name__)

@router.get('/')
def hello():
    rtspUrl = "rtsp://user:js1111111@10.13.10.3:7001/000F7C175E88"
    cam = Camera(rtspUrl)
    return Response(cam.boxModelDetect(AiModelPath.yolov8n, []), mimetype='multipart/x-mixed-replace; boundary=frame')

@router.get('/2')
def hello2():
    rtspUrl = "rtsp://user:js1111111@10.13.10.3:7001/000F7C175E88"
    cam = Camera(rtspUrl)
    return Response(cam.test2(AiModelPath.yolov8n), mimetype='multipart/x-mixed-replace; boundary=frame')