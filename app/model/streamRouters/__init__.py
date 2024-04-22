from flask import Blueprint, request, Response, jsonify
from app.module.rtspStream import Camera
from app.config import AiModelPath

streamRouter = Blueprint('streamRouter', __name__)

"""
/stream?rtspUrl=rtsp://user:js1111111@10.13.10.3:7001/000F7C175E88&thickness=10&textThickness=10&textScale=10

http://127.0.0.1:5000/stream/detection?rtspUrl=rtsp://user:js1111111@10.13.10.3:7001/000F7C175E88

rtsp://10.13.10.54/rtsp_tunnel?p=0&line=1&inst=1&aon=1&aud=1&vcd=2
rtsp://service:Js!152604@10.13.10.54/rtsp_tunnel

http://user:js1111111@10.13.10.3:7001/media/1839f6ea-00eb-80fd-5887-b6c79e28e1ca.mp4?pos=1711554600000&endPos=1711554603000
"""

@streamRouter.get('')
def stream():
    rtspUrl = request.args.get('rtspUrl', None)
    if rtspUrl == None: return jsonify(message="rtsp url not found."), 400
    cam = Camera(rtspUrl)
    return Response(cam.genFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@streamRouter.get('/<model>')
def model(model):
    rtspUrl = request.args.get('rtspUrl', None)
    if rtspUrl == None: return jsonify(message="rtsp url not found."), 400
    
    classList = request.args.get('classList', []) # 預測框線粗度
    thickness = int(request.args.get('thickness', 2)) # 文字粗度
    textThickness = int(request.args.get('textThickness', 2)) # 文字粗度
    textScale = int(request.args.get('textScale', 1)) # 文字大小
    confidence = float(request.args.get('confidence', 0)) #模型偵測的閥值
    
    # 轉換class清單
    if len(classList) != 0: 
        classList = classList.split(',')
        classList = [int(x) for x in classList]

    # Camera obj
    cam = Camera(rtspUrl)

    # 普通標註預測
    if model == 'detection':
        return Response(cam.boxModelDetect(AiModelPath.yolov8n, classList, thickness, textThickness, textScale, confidence), mimetype='multipart/x-mixed-replace; boundary=frame'), 200
    
    # 越線偵測計數
    if model == 'lineCounter':
        lineStartX = int(request.args.get('lineStartX', 2))
        lineStartY = int(request.args.get('lineStartY', 2))
        lineEndX = int(request.args.get('lineEndX', 1))
        lineEndY = int(request.args.get('lineEndY', 1))
        return Response(cam.lineModelDetect(AiModelPath.yolov8n, classList, lineStartX, lineStartY, lineEndX, lineEndY, confidence), mimetype='multipart/x-mixed-replace; boundary=frame'), 200

    # 範圍偵測計數
    if model == 'countInRange':
        return Response(cam.countObjectsInRange(AiModelPath.yolov8n, classList=classList, confidence=confidence), mimetype='multipart/x-mixed-replace; boundary=frame'), 200

    # 車牌辨識
    if model == 'licensePlate':
        return Response(cam.licensePlateRecognition(), mimetype='multipart/x-mixed-replace; boundary=frame'), 200
    
    # 摩托車偵測
    if model == 'motorcycle':
        return Response(cam.boxModelDetect(AiModelPath.motorcycle, [], thickness, textThickness, textScale, confidence), mimetype='multipart/x-mixed-replace; boundary=frame'), 200
    
    return jsonify(message='not found model...'), 400