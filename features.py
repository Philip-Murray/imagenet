import loader
import numpy as np
from   loader import AsciImage



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
    mapping = [[charmap(c) for c in img_row] for img_row in img.pixels]


    x_div = 5
    y_div = 5
    x_boxamt = int(loader.faceDataset.dim_x / x_div)
    y_boxamt = int(loader.faceDataset.dim_y / y_div)

    if  ffe_onetime.ffe_onetime:
        ffe_onetime.ffe_onetime = False
        faces.dim_x = x_boxamt
        faces.dim_y = y_boxamt


    tiles = []

    for ybox in range(y_boxamt):
        tiles.append([])
        for xbox in range(x_boxamt):
            sum = 0
            for y in range(ybox*y_div, (1+ybox)*y_div):
                for x in range(xbox*x_div, (1+xbox)*x_div):
                    sum += mapping[y][x]
            if(sum > 3):
                sum = 1
            else:
                sum = 0

            #avg = min(sum, 9)
            tiles[ybox].append(sum)

    return [aggregate for boxed_row in tiles for aggregate in boxed_row] #Must be same size for any img in face set

class ffe_onetime:
    ffe_onetime = True





def printNumpySubset(nparray, index, dim_x, dim_y):
    for i in range(dim_y):
        print(nparray[index, i*dim_x : (i+1)*dim_x])





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
            printNumpySubset(mnist.X_train, index, mnist.dim_x, mnist.dim_y)
        if set == "valid" or set == 1:
            printNumpySubset(mnist.X_valid, index, mnist.dim_x, mnist.dim_y)
        if set == "test" or set == 2:
            printNumpySubset(mnist.X_test,  index, mnist.dim_x, mnist.dim_y)




class faces:
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
            printNumpySubset(faces.X_train, index, faces.dim_x, faces.dim_y)
        if set == "valid" or set == 1:
            printNumpySubset(faces.X_valid, index, faces.dim_x, faces.dim_y)
        if set == "test" or set == 2:
            printNumpySubset(faces.X_test,  index, faces.dim_x, faces.dim_y)


def init(): #Has hardcode hardcoded value

    mnist.X_train = np.array([ mnistFeatureExtraction(img) for img in loader.mnistDataset.training_data])
    mnist.X_valid = np.array([ mnistFeatureExtraction(img) for img in loader.mnistDataset.validation_data])
    mnist.X_test  = np.array([ mnistFeatureExtraction(img) for img in loader.mnistDataset.test_data])

    #mnist.Y_train = np.array([ [1 if x==img.label else 0 for x in range(10)] for img in loader.mnistDataset.training_data])
    #mnist.Y_valid = np.array([ [1 if x==img.label else 0 for x in range(10)] for img in loader.mnistDataset.validation_data])
    #mnist.Y_test  = np.array([ [1 if x==img.label else 0 for x in range(10)] for img in loader.mnistDataset.test_data])

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








