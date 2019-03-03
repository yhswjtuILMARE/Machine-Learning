'''
Created By ilmare
@date 2019-2-25
'''
from Lib.AutoEncoder import AutoEncoder
from Tools.Detector import PhotoParser

modelFile = r"/home/ilmare/Desktop/FaceReplace/shape_predictor_68_face_landmarks.dat"

if __name__ == "__main__":
    videoPath = r"/home/ilmare/Desktop/FaceReplace/data/video/source.mp4"
    parser = PhotoParser(videoPath, modelFile, (128, 128))
    obj = AutoEncoder(0.01, 100, 64, 3)
    obj.train(parser.trainImagePath)