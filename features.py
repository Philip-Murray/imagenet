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
    return [charmap(pixel) for img_row in img.pixels for pixel in img_row] #Must be same size for any img in mnist set
    

def faceFeatureExtraction(img: AsciImage):
    def charmap(c: str):
        if c == " ":
            return 0
        if c == "#":
            return 1
    return [charmap(pixel) for img_row in img.pixels for pixel in img_row] #Must be same size for any img in face set



class mnist:
    featureVectorSize = None

    X_train = None
    Y_train = None

    X_valid = None
    Y_valid = None

    X_test = None
    Y_test = None
        

class faces:
    featureVectorSize = None

    X_train = None
    Y_train = None

    X_valid = None
    Y_valid = None

    X_test = None
    Y_test = None


def init():

    mnist.X_train = np.array([ mnistFeatureExtraction(img) for img in loader.mnistDataset.training_data])
    mnist.X_valid = np.array([ mnistFeatureExtraction(img) for img in loader.mnistDataset.validation_data])
    mnist.X_test  = np.array([ mnistFeatureExtraction(img) for img in loader.mnistDataset.test_data])

    mnist.Y_train = np.array([ img.label for img in loader.mnistDataset.training_data])
    mnist.Y_valid = np.array([ img.label for img in loader.mnistDataset.validation_data])
    mnist.Y_test  = np.array([ img.label for img in loader.mnistDataset.test_data])

    _, mnist.featureVectorSize = mnist.X_train.shape


    faces.X_train = np.array([ faceFeatureExtraction(img) for img in loader.faceDataset.training_data])
    faces.X_valid = np.array([ faceFeatureExtraction(img) for img in loader.faceDataset.validation_data])
    faces.X_test  = np.array([ faceFeatureExtraction(img) for img in loader.faceDataset.test_data])

    faces.Y_train = np.array([ img.label for img in loader.faceDataset.training_data])
    faces.Y_valid = np.array([ img.label for img in loader.faceDataset.validation_data])
    faces.Y_test  = np.array([ img.label for img in loader.faceDataset.test_data])
    
    _, faces.featureVectorSize = faces.X_train.shape

    if False:
        print(mnistDataset.X_train.shape)
        print(mnistDataset.X_valid.shape)
        print(mnistDataset.X_test.shape)

        print(faceDataset.X_train.shape)
        print(faceDataset.X_valid.shape)
        print(faceDataset.X_test.shape)

        print(faceDataset.featureVectorSize)
    
    #for i in range(28):
        #print(list[0, i*(28) : (i+1)*28])







