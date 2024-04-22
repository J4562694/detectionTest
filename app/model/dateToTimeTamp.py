import datetime

# 將標準時間轉換成紀元時間
def dateToTimeTamp(dateTime):
    dateList = [int(dateTime[0:4]), int(dateTime[5:7]), int(dateTime[8:10]), int(dateTime[11:13]), int(dateTime[14:16]), int(dateTime[17:19])]
    epochTime = int(datetime.datetime(*dateList).timestamp())
    return epochTime