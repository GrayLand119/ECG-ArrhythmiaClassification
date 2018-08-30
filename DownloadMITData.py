import requests
import os
import threading


class DownloadThread(threading.Thread):
    downloadFinishCallBack: callable
    url: str
    fileName: str
    outputPath: str

    def __init__(self, url, fileName, outputPath, downloadFinishCallBack):
        threading.Thread.__init__(self)
        self.url = url
        self.fileName = fileName
        self.outputPath = outputPath
        self.downloadFinishCallBack = downloadFinishCallBack

    def run(self):
        print("Start Download...", self.url)
        req = requests.get(self.url)
        count = 0
        with open(self.outputPath + self.fileName, 'wb') as f:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    count += 1
                    print("Download...{0}KB".format(count))
                    f.write(chunk)

        print("Finish Download...:", self.url)
        if self.downloadFinishCallBack is not None:
            self.downloadFinishCallBack()


class DownloadManager:
    outputPath = ''
    EXTENSION_ATR = 'atr'
    EXTENSION_DAT = 'dat'
    EXTENSION_HEA = 'hea'
    netPath = 'https://www.physionet.org/physiobank/database/mitdb/'
    fileNames = []
    concurrenceNum = 0
    allThreads = []

    def startDownLoad(self):
        trainFileNames = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124]
        testFilesNames = [200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234]
        self.fileNames = trainFileNames + testFilesNames
        # self.fileNames = [102, 104, 107, 110, 120, 204, 206, 211, 216, 217, 218, 224, 225, 226, 227, 229]
        self.fileNames = [229]
        print(self.fileNames)
        # 217
        curPath = os.getcwd()
        self.outputPath = curPath + '/MIT-DB-Download/'
        if not os.path.exists(self.outputPath):
            os.makedirs(self.outputPath)

        print('OUT_PUT_PATH=', self.outputPath)

        self.autoDownLoad()

    def autoDownLoad(self):
        self.concurrenceNum -= 1
        if self.concurrenceNum > 4:
            return

        if len(self.fileNames) > 0:
            fn = self.fileNames.pop()
            fn = str(fn)
            fileUrl1 = self.netPath + fn + '.' + self.EXTENSION_ATR
            fileUrl2 = self.netPath + fn + '.' + self.EXTENSION_DAT
            fileUrl3 = self.netPath + fn + '.' + self.EXTENSION_HEA
            th1 = DownloadThread(fileUrl1, fn + '.' + self.EXTENSION_ATR, self.outputPath, self.autoDownLoad)
            th2 = DownloadThread(fileUrl2, fn + '.' + self.EXTENSION_DAT, self.outputPath, self.autoDownLoad)
            th3 = DownloadThread(fileUrl3, fn + '.' + self.EXTENSION_HEA, self.outputPath, self.autoDownLoad)
            th1.start()
            th2.start()
            th3.start()
            self.allThreads.extend([th1, th2, th3])
            # self.allThreads.extend([th1])
            self.concurrenceNum += 1
        pass

if __name__ == '__main__':
    downloadManager = DownloadManager()
    downloadManager.startDownLoad()

