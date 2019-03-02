'''
Created By ILMARE
@Date 2019-3-2
'''

from dlib import shape_predictor as predictor
from dlib import get_frontal_face_detector as detector
import cv2
import os

modelFile = r"/home/ilmare/Desktop/FaceReplace/shape_predictor_68_face_landmarks.dat"

class PhotoParser:
    def __init__(self, videoPath, savePath, modelFile, trainPath, destShape):
        self._modelFile = modelFile
        self._videoPath = videoPath
        self._savePath = savePath
        self._trainPath = trainPath
        self._destShape = destShape
        self._photoCount = 0
    def getPhotoFromVideo(self):
        vc = cv2.VideoCapture(self._videoPath)
        while True:
            rval, frame = vc.read()
            if rval:
                cv2.imwrite("{0}{1}.jpg".format(self._savePath, self._photoCount), frame)
                self._photoCount += 1
                if (self._photoCount % 100) == 0:
                    print("Total Parsed Photo: ", self._photoCount)
            else:
                break
        print("Total Parsed Photo: ", self._photoCount)
        vc.release()
    def detectorPhotoFace(self):
        detectorObj = detector()
        try:
            fileList = os.listdir(self._savePath)
            for file, idx in zip(fileList, range(len(fileList))):
                filePath = "{0}{1}".format(self._savePath, file)
                img = cv2.imread(filePath)
                rects = detectorObj(img, 1)
                if len(rects) > 1 or len(rects) == 0:
                    print(filePath)
                    continue
                rect = rects[0]
                left, top, width, height = rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top()
                img = img[top:top + height, left:left + width, :]
                img = self.__resizePhoto(img)
                cv2.imwrite("{0}{1}.jpg".format(self._trainPath, idx), img)
                if (idx % 100) == 0:
                    print("current index: ", idx)
        except Exception as e:
            print(e)
    def __resizePhoto(self, img):
        try:
            height = img.shape[0]
            width = img.shape[1]
            interval = abs(width - height)
            margin = interval // 2
            if width > height:
                if (interval % 2) == 0:
                    img = img[:, margin: width - margin, :]
                else:
                    img = img[:, margin + 1: width - margin, :]
            elif height > width:
                if (interval % 2) == 0:
                    img = img[margin: height - margin, :, :]
                else:
                    img = img[margin + 1: height - margin, :, :]
            return cv2.resize(img, self._destShape)
        except Exception as e:
            print(e)
            return None


if __name__ == "__main__":
    videoPath = r"/home/ilmare/Desktop/FaceReplace/data/video/source.mp4"
    savePath = r"/home/ilmare/Desktop/FaceReplace/data/img/"
    trainPath = r"/home/ilmare/Desktop/FaceReplace/data/train/"
    obj = PhotoParser(videoPath, savePath, modelFile, trainPath, (128, 128))
    obj.detectorPhotoFace()


