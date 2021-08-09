from loader import AsciImage

import numpy

def mnistFeatureExtraction(img: AsciImage):
    flat_pixels = [pixel for img_row in img.pixels for pixel in img.pixels]
    def charmap(c: str):
        if c == " ":
            return 0
        if c == "#":
            return 1
        if c == "+":
            return 1
    return map(lambda c: charmap(c), flat_pixels) #Must be same size for any img in mnist set
    

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


def mnistProcess():
    

def init():
    pass






