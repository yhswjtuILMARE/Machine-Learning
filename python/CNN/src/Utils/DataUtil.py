'''
Created on 2018年7月2日

@author: IL MARE
'''
import os
import numpy as np
import re
from PIL import Image, ImageEnhance
import time
import random

file_path = r"G:/研究生课件/人工神经网络/神经网络/dataset_cat_dog_classification/dataset/"

class ImageObject:
    def __init__(self, filePath, shape=(224, 224)):
        self._shape = shape
        self._filePath = filePath
        self.generateDataSet()
    def generateDataSet(self):
        list = os.listdir(self._filePath)
        self._train_path = "{0}{1}".format(self._filePath, "train/")
        self._test_path = "{0}{1}".format(self._filePath, "test/")
        if os.listdir(self._train_path) and os.listdir(self._test_path):
            self._trainSet = set(os.listdir(self._train_path))
            self._testSet = set(os.listdir(self._test_path))
            return
        print("正在初始化训练集和测试集。。。")
        self._trainSet = set()
        self._testSet = set(list) - set(["train", "test"])
        for i in range(len(list)):
            if i % 500 == 0:
                print(i)
            index = np.random.randint(0, len(list), 1)[0]
            item = list[index]
            if item == "test" or item == "train":
                continue
            if item not in self._trainSet:
                self._trainSet.add(item)
                image = Image.open("{0}{1}".format(file_path, item))
                image = image.resize(self._shape)
                image.save("{0}{1}".format(self._train_path, item))
        self._testSet = self._testSet - self._trainSet
        i = 0
        for name in self._testSet:
            i += 1
            if i % 500 == 0:
                print(i)
            image = Image.open("{0}{1}".format(file_path, name))
            image = image.resize(self._shape)
            image.save("{0}{1}".format(self._test_path, name))
    def nextBatch(self, num=50):
        list = random.sample(self._trainSet, num)
        train = []
        label = []
        for name in list:
            image = Image.open("{0}{1}".format(self._train_path, name))
            train.append(np.asarray(image))
            if re.match(r"^cat.*$", name):
                label.append(np.array([1, 0]))
            else:
                label.append(np.array([0, 1]))
        return np.array(train), np.array(label)
    def generateTestBatch(self, num=100):
        test = []
        label = []
        for name in self._testSet:
            image = Image.open("{0}{1}".format(self._test_path, name))
            test.append(np.asarray(image))
            if re.match(r"^cat.*$", name):
                label.append(np.array([1, 0]))
            else:
                label.append(np.array([0, 1]))
            if len(test) % num == 0:
                yield np.array(test), np.array(label)
                test = []
                label = []
        yield np.array(test), np.array(label)

if __name__ == "__main__":
    start = time.clock()
    obj = ImageObject(file_path)
    train, label = obj.nextBatch(50)
    print(train.shape, label.shape)
    for test, testLabel in obj.generateTestBatch(126):
        print(test.shape, testLabel.shape)
    print(time.clock() - start)