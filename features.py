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
    

class faceConversionVars:
    x_div = 10
    y_div = 10

    x_boxamt = None
    y_boxamt = None

    def setup():
        fcv = faceConversionVars
        fcv.x_boxamt = int(loader.faceDataset.dim_x / fcv.x_div)
        fcv.y_boxamt = int(loader.faceDataset.dim_y / fcv.y_div)

    is_setup = False


def faceFeatureExtraction(img: AsciImage):
    
    if not faceConversionVars.is_setup:
        faceConversionVars.setup()
    
    def charmap(c: str):
        if c == " ":
            return 0
        if c == "#":
            return 1
    mapping = [[charmap(c) for c in img_row] for img_row in img.pixels]

    x_div = 10
    y_div = 10
    x_boxamt = int(loader.faceDataset.dim_x / x_div)
    y_boxamt = int(loader.faceDataset.dim_y / y_div)

    tiles = []
    

    for ybox in range(y_boxamt):
        tiles.append([])
        for xbox in range(x_boxamt):
            sum = 0
            for y in range(ybox*y_div, (1+ybox)*y_div):
                for x in range(xbox*x_div, (1+xbox)*x_div):
                    sum += mapping[y][x]
            avg = sum / (y_div*x_div)
            tiles[ybox].append(avg)


    return [aggregate for boxed_row in tiles for aggregate in boxed_row] #Must be same size for any img in face set





class mnist:
    featureVectorSize = None
    dim_x = None
    dim_y = None

    X_train = None
    Y_train = None

    X_valid = None
    Y_valid = None

    X_test = None
    Y_test = None

    def printImage(index: int, set="train"):
        if set == "train" or set == 0:

        
        

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

    if True:
        print(mnist.X_train.shape)
        print(mnist.X_valid.shape)
        print(mnist.X_test.shape)

        print(faces.X_train.shape)
        print(faces.X_valid.shape)
        print(faces.X_test.shape)

        print(faces.featureVectorSize)
    
    #for i in range(28):
        #print(list[0, i*(28) : (i+1)*28])







