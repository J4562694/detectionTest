from app.module.sqlQuery.conn import IMAGE_RECOGNITION_POOL

class DetectRemarkQuery:

    # 新增攝影機資料
    @staticmethod
    def writeCameraInformation(obj):
        sql = f"""
            INSERT INTO
                camera(title, rtspUrl, nxDeviceId)
            VALUES
                ({obj["title"]}, {obj["rtspUrl"]}, {obj["nxDeviceId"]});
        """
        IMAGE_RECOGNITION_POOL.doSql(sql)
    
    # 新增預測紀錄時間戳
    @staticmethod
    def addPredictTimestamp(obj):
        """
        ### 新增預測紀錄時間戳
        args:
        - cameraId: 攝影機編號
        - predictTime: 紀錄時間戳
        """
        sql = f"""
            INSERT INTO
                predictTimestamp(cameraId, predictTime)
            VALUES
                ({obj['cameraId']}, '{obj['predictTime']}');
        """
        IMAGE_RECOGNITION_POOL.doSql(sql)

    # 新增預測紀錄
    @staticmethod
    def addPrediction(obj):
        """
        ### 新增預測紀錄
        args:
        - cameraId: 攝影機編號
        - predictTime: 紀錄時間戳
        - label: 類別名稱
        - boundingBox: 邊框位址(x1,y1,x2,y2)
        - confidence: 信心指數
        """
        sql = f"""
            INSERT INTO 
                prediction(cameraId, predictTime,label, boundingBox, confidence)
            VALUES
                ({obj['cameraId']}, '{obj['predictTime']}', '{obj['label']}', '{obj['boundingBox']}', {obj['confidence']});
        """
        IMAGE_RECOGNITION_POOL.doSql(sql)
    
    # 取得所有攝影
    @staticmethod
    def getAllCamera() -> list:
        sql = """
            SELECT
                id,
                title,
                rtspUrl,
                nxDeviceId
            FROM
                camera;
        """
        rows = IMAGE_RECOGNITION_POOL.selectSql(sql)
        return rows
    
    # 繪製偵測框取得該時間內的資料
    def getPredictionData(predictTime):
        sql = f"""
            SELECT
                DATE_FORMAT(predictTime, '%Y-%m-%d %H:%i:%s') as predictTime,
                label,
                boundingBox,
                confidence
            FROM
                prediction
            WHERE
                predictTime >= '{predictTime}'
            AND
                boundingBox != 0;
        """

        rows = IMAGE_RECOGNITION_POOL.selectSql(sql)

        newData = []
        timestamp = ''
        newDataIndex = 0
        for index, row in enumerate(rows):
            
            if index == 0:
                timestamp = row['predictTime']
                newData.append([])
            
            if timestamp == row['predictTime']:
                newData[newDataIndex].append(row)
            else:
                timestamp = row['predictTime']
                newData.append([])
                newDataIndex += 1

        print(newData)
        return newData