from flask import Blueprint, request, Response, jsonify, send_file
from app.module.dateToTimeTamp import dateToTimeTamp
from app.module.videoDownload import videoDownload
from app.module.rtspStream import Camera
from app.config import AiModelPath, downloadVideoPath
from app.module.sqlQuery.detectRemarkQuery import DetectRemarkQuery
import requests
import os

recordViewRouter = Blueprint("recordViewRouter", __name__)

# 單純回放影片路由
@recordViewRouter.get('')
def recordView():
    # http://10.13.10.58:5000/recordView?videoSeconds=10&videoPos=2024-03-27T08:55:50&videoEndPos=2024-03-27T08:56:50&videoFormat=mp4&&modelName=yolov8n
    
    videoSeconds = request.args.get('videoSeconds', 5)          #預設沒有影片秒數就5秒
    videoPos  = request.args.get('videoPos')                    #影片開始日期
    videoEndPos = request.args.get('videoEndPos')
    videoFormat = str(request.args.get('videoFormat', 'mp4'))    #影片格式
    modelName = str(request.args.get('modelName'))              #模型名稱
    modelPath = getattr(AiModelPath, modelName, None)

    posStamp = (dateToTimeTamp(videoPos))*1000
    endPosStamp = (dateToTimeTamp(videoEndPos))*1000

    videoUrl = f"http://user:js1111111@10.13.10.3:7001/media/1839f6ea-00eb-80fd-5887-b6c79e28e1ca.{videoFormat}?pos={posStamp}&endPos={endPosStamp}&duration={videoSeconds}"

    cam = Camera(videoUrl)
    try:
        response = requests.get(videoUrl, stream=True)
        if response.status_code == 200:
            return Response(cam.boxModelDetect(modelPath), mimetype='multipart/x-mixed-replace; boundary=frame'), 200
        else:
            return jsonify(message="Failed to download the video."), response.status_code
    except requests.RequestException as e:
        print(e)
        return jsonify(message="Error while downloading the video."), 500

# 下載影片路由
@recordViewRouter.get('/download')
def videoDownloadUrl():
    videoSeconds = request.args.get('videoSeconds', 5)          #預設沒有影片秒數就5秒
    videoPos  = request.args.get('videoPos')                    #影片開始日期
    videoFormat = str(request.args.get('videoFormat', 'ts'))    #影片格式
    posStamp = (dateToTimeTamp(videoPos))*1000
    modelName = str(request.args.get('modelName'))              #模型名稱
    modelPath = getattr(AiModelPath, modelName, None)

    videoUrl = f"http://user:js1111111@10.13.10.3:7001/hls/1839f6ea-00eb-80fd-5887-b6c79e28e1ca.{videoFormat}?pos={posStamp}&duration={videoSeconds}&hi=true"
    videoName = videoDownload(videoUrl)

    cam = Camera(f'{downloadVideoPath.videoPath}{videoName}')

    checkData = cam.videoPredict(modelPath, filePath=f'{downloadVideoPath.videoPath}\\{videoName}')
    os.remove(f'{downloadVideoPath.videoPath}\\{videoName}')

    if checkData == True:
        return send_file(f"{downloadVideoPath.videoPath}\\detect.mp4",  as_attachment=True), 200
    else:
        return jsonify("發生錯誤"), 400
    
# 回放事件路由
@recordViewRouter.get('/event')
def eventReplay():
    videoPos = request.args.get('videoPos')
    videoEndPos = request.args.get('videoEndPos')
    predictTime = request.args.get('predictTime')
    results = DetectRemarkQuery.getPredictionData(predictTime)
    videoUrl = f"http://user:js1111111@10.13.10.3:7001/media/1839f6ea-00eb-80fd-5887-b6c79e28e1ca.mp4?pos={predictTime}&endPos=2024-04-11 13:41:08"
    cam = Camera(videoUrl)
    
    try:
        response = requests.get(videoUrl, stream=True)
        if response.status_code == 200:
            return Response(cam.drawDetectionBoxes(detectionData=results), mimetype='multipart/x-mixed-replace; boundary=frame'), 200
        else:
            return jsonify(message="Failed to download the video."), response.status_code
    except requests.RequestException as e:
        print(e)
        return jsonify(message="Error while downloading the video."), 500