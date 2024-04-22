from datetime import datetime
import requests

# 下載影片並以日期為檔名
def videoDownload(videoUrl):
    time = datetime.now()
    localTime = str(time)
    videoName = localTime[0:10] + "-" + localTime[11:13] + "-" + localTime[14:16] + ".ts"
    response = requests.get(videoUrl, stream=True)
    if response.status_code == 200:
        with open(f'./videoDownload/{videoName}', 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
    else:
        print("error")

    return videoName
