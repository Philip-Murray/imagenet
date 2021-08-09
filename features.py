from loader import AsciImage
import loader

import numpy as np

def mnistFeatureExtraction(img: AsciImage):
    def charmap(c: str):
        if c == " ":
            return 0
        if c == "#":
            return 1
        if c == "+":
            return 1
    return [charmap(pixel) for img_row in img.pixels for pixel in img.pixels] #Must be same size for any img in mnist set
    

def faceFeatureExtraction(img: AsciImage):
    flat_pixels = [pixel for img_row in img.pixels for pixel in img.pixels]
    def charmap(c: str):
        if c == " ":
            return 0
        if c == "#":
            return 1
        if c == "+":
            return 1
    return map(lambda c: charmap(c), flat_pixels) #Must be same size for any img in face set



class mnistDataset:
    featureLength = None

    X_train = None
    Y_train = None

    X_valid = None
    Y_valid = None

    X_test = None
    Y_test = None
        

class faceDataset:
    featureLength = None

    X_train = None
    Y_train = None

    X_valid = None
    Y_valid = None

    X_test = None
    Y_test = None


def mnistFeaturesInit():
    list = [ mnistFeatureExtraction(img) for img in loader.mnistDataset.test_data]
    
    print(len(list))
    print(len(list[0]))

    for i in range(28):
        print(list[i:i, 0:28])

def init():
    pass






