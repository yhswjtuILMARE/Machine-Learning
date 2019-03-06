'''
Created By ILMARE
@Date 2019-3-3
'''
import re
import os
import numpy as np
import cv2

class ImageTrainObject:
    def __init__(self, filePath, batchSize):
        self._filePath = filePath
        self._batchSize = batchSize
        # if re.match(r"^/.+/[^.]+$", self._filePath) is None:
        #     raise Exception("filePath is invalid")
        if self._filePath[len(self._filePath) - 1] != '/':
            self._filePath += '/'
        self._fileItems = os.listdir(self._filePath)
        if batchSize >= self.DataCount:
            raise Exception("Too big batchSize")
    @property
    def DataCount(self):
        return len(self._fileItems)
    def generateBatch(self):
        beginIdx = np.random.randint(0, self.DataCount - self._batchSize)
        destFile = self._fileItems[beginIdx: beginIdx + self._batchSize]
        return_mat = []
        for file in destFile:
            img = cv2.imread("{0}{1}".format(self._filePath, file))
            return_mat.append(img)
        return np.array(return_mat, dtype=np.float32)

if __name__ == "__main__":
    filePath = r"F:/tensorflow/automodel/scrawler/video/trainImg/"
    batchSize = 64
    obj = ImageTrainObject(filePath, batchSize)
    obj.generateBatch()
    print(obj.DataCount)