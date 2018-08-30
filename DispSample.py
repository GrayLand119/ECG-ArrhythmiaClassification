from GLTools import *
import os
import wfdb as wf
import wfdb.processing


if __name__ == '__main__':

    fileNum = 100
    fileName = str(fileNum)

    items = GLTools.getItemsInDir(os.getcwd() + '/MIT-DB')
    print(items)


