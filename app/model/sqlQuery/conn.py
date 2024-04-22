from typing import Any
from mysql import connector
from app.config import SqlStuff

class MySqlPool:

    # 設定連線資料
    def __init__(self, database: str) -> None:
        self.host = SqlStuff.host
        self.user = SqlStuff.user
        self.passowrd = SqlStuff.password
        self.database = database
    
    # 取得資料庫連線
    def getConn(self):
        return connector.connect(
            host=self.host,
            user=self.user,
            password=self.passowrd,
            database=self.database
        )

    # insert, update, delete: 輸入sql query並執行它
    def doSql(self, sql: str):
        conn = self.getConn()
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
        conn.close()

    # select: sql query並返回查詢結果
    def selectSql(self, sql: str):
        conn = self.getConn()
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [col[0] for col in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return rows
    
    # 可以對CRUD同時操作
    def executeAndFetchSql(self, doSql: str, selectSql:str):
        conn = self.getConn()
        try:
            cursor = conn.cursor()
            cursor.execute(doSql)
            conn.commit()

            cursor.execute(selectSql)
            columns = [col[0] for col in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            return rows
        except:
            pass
        finally:
            conn.close()

IMAGE_RECOGNITION_POOL = MySqlPool(database='image_recognition')
