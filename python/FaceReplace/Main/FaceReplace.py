'''
Created By ilmare
@date 2019-2-25
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
from dlib import get_frontal_face_detector as detector
from dlib import shape_predictor as predictor

def detectFaceWithRectangle(img):
    '''
    该函数用于检测照片中的人脸，并且返回人脸所在的矩形坐标
    :param img:
    :return:
    '''
    detector_obj = detector()
    rect = detector_obj(img, 1)
    return_mat = []
    for item in rect:
        return_mat.append([item.left(), item.top(),
                           item.right() - item.left(),
                           item.top() - item.bottom()])
    return np.asarray(return_mat, dtype=np.float32), rect

def getFaceLandMark(img, rect, modelFile):
    '''
    该函数用于标记面部的68个关键点返回一个68*2矩阵以及shape_predictor的原始返回值
    :param img:
    :param rect:
    :param modelFile:
    :return:
    '''
    if modelFile == None:
        raise Exception("modelFile is absent")
    predictor_obj = predictor(modelFile)
    points = predictor_obj(img, rect)
    landMarkPoints = []
    for p in points.parts():
        landMarkPoints.append([p.x, p.y])
    return np.asarray(landMarkPoints, dtype=np.float32), points

def tmp_plot():
    fig = plt.figure("test")
    ax = fig.add_subplot(121)
    ax.imshow(sourceImage)
    for item in sourceShape:
        print(item)
        rect = plt.Rectangle((item[0], item[1]),
                             item[2], -item[3], color='r', fill=False)
        ax.add_patch(rect)
    ax.plot(sourcePoints[:, 0], sourcePoints[:, 1], "r.")
    bx = fig.add_subplot(122)
    bx.imshow(destImage)
    for item in destShape:
        print(item)
        rect = plt.Rectangle((item[0], item[1]),
                             item[2], -item[3], color='r', fill=False)
        bx.add_patch(rect)
    bx.plot(destPoints[:, 0], destPoints[:, 1], "r.")
    plt.show()

def transformationFormPoints(sourcePoints, destPoints):
    sourcePoints = np.asmatrix(sourcePoints, dtype=np.float32)
    destPoints = np.asmatrix(destPoints, dtype=np.float32)
    sourceMean = np.mean(sourcePoints, 0)
    destMean = np.mean(destPoints, 0)
    sourcePoints -= sourceMean
    destPoints -= destMean
    sourceStd = np.std(sourcePoints)
    destStd = np.std(destPoints)
    sourcePoints /= sourceStd
    destPoints /= destStd
    U, S, Vt = np.linalg.svd(destPoints.T * sourcePoints)
    R = (U * Vt).T
    return np.vstack([np.hstack(((sourceStd / destStd) * R,
                                 sourceMean.T - (sourceStd / destStd) * R * destMean.T)),
                      np.matrix([0., 0., 1.])])

def fig(img):
    fig = plt.figure("test")
    ax = fig.add_subplot(111)
    ax.imshow(img)
    plt.show()


if __name__ == "__main__":
    dest = r"/home/ilmare/Desktop/FaceReplace/dest.jpg"
    source = r"/home/ilmare/Desktop/FaceReplace/source.jpg"
    modelFile = r"/home/ilmare/Desktop/FaceReplace/shape_predictor_68_face_landmarks.dat"
    sourceImage = cv2.imread(source)
    destImage = cv2.imread(dest)
    sourceShape, sourceRects = detectFaceWithRectangle(sourceImage)
    destShape, destRects = detectFaceWithRectangle(destImage)
    sourcePoints, _ = getFaceLandMark(sourceImage, sourceRects[0], modelFile)
    destPoints, _ = getFaceLandMark(destImage, destRects[0], modelFile)
    p_matrix = transformationFormPoints(sourcePoints, destPoints)
    outputSize = destImage.shape
    outputMatrix = np.zeros(outputSize, dtype=destImage.dtype)
    print(destImage.dtype)
    cv2.warpAffine(src=sourceImage, dst=outputMatrix, M=p_matrix[:2],
                   dsize=(outputSize[1], outputSize[0]),
                   borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)
    cv2.imshow("girl", outputMatrix)
    cv2.waitKey(0)