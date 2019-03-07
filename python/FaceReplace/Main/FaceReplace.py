'''
Created By ilmare
@date 2019-2-25
'''
import sys
import os

sys.path.append(os.getcwd())

from Tools.Detector import PhotoParser
from Lib.AutoEncoder import AutoEncoder

modelFile = r"/home/ilmare/Desktop/FaceReplace/shape_predictor_68_face_landmarks.dat"

if __name__ == "__main__":
    videoPath = r"/home/yanghang/faceswap/video/source.mp4"
    parser = PhotoParser(videoPath, modelFile, (128, 128))
    obj = AutoEncoder(0.005, 400, 1, "/disk/model/", 3)
    obj.train(parser.trainImagePath)