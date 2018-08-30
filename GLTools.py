import os


class GLTools (object):

    @staticmethod
    def curPath() -> str:
        return os.getcwd()

    @staticmethod
    def curPathWithItems(items: list) -> str:
        path = GlobalTool.curPath()
        if type(items) == list:
            for i in range(len(items)):
                path += "/{0}/".format(items[i])
        elif type(items) == str:
            path += "/{0}/".format(items)

        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @staticmethod
    def safePath(path) -> str:
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @staticmethod
    def getFileName(filePath: str = "", withExtension: bool = True) -> str:
        if len(filePath) == 0:
            return ''
        else:
            tArr = filePath.rsplit('/', 1)
            fileName = tArr.pop()
            if withExtension:
                return fileName
            else:
                tArr2 = fileName.rsplit('.', 1)
                return tArr2[0]

    # 获取文件夹下的文件和目录
    @staticmethod
    def getItemsInDir(directory: str, flags: int = 0) -> list:
        """
        :param directory: 将要列出文件的目录
        :param flags: 标记, 0 - 列出全部, 1 - 文件, 2 - 目录
        :return: 根据标记返回对应的items
        """
        try:
            allfiles = os.listdir(directory)
            dirs = []
            files = []
            for item in allfiles:
                tItemPath = directory + '/' + item
                if os.path.isfile(tItemPath):
                    files.append(tItemPath)
                else:
                    dirs.append(tItemPath)
            if flags == 0:
                dirs.extend(files)
                return dirs
            elif flags == 1:
                return files
            else:
                return dirs
        except Exception as e:
            print(str(e))
            return []