# Dev config
import os

class SqlStuff:
    host = "10.13.10.14"
    user = "jingzesystem"
    password = "js152604"

class RedisPath:
    host = "10.13.10.5"
    port = 6379
    db = 1
    password = ""

class AiModelPath:
    faceMask = './models/face-mask.pt'
    licensePlateDetector = './models/license-plate-detector.pt'
    wearingHelmet = './models/wearing-helmet.pt'
    yolov8n = './models/yolov8n.pt'
    motorcycle = './models/motorcycle.pt'

class videoPath:
    videoPath = './video/'

class downloadVideoPath:
    videoPath = f'{os.path.join(os.getcwd(), "videoDownload")}'