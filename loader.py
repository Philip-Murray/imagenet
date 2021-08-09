import os, sys

A4_PATH = os.path.dirname(os.path.realpath(__file__))

class AsciImage:
    def __init__(self, array2d, label: int):
        self.pixels = array2d
        self.label  = label



class mnistDataset: #namespace 
    training_data = []
    training_labels = []

    validation_data = []
    validation_labels = []

    test_data = []
    test_labels = []

class faceDataset: #namespace
    training_data = []
    training_labels = []

    validation_data = []
    validation_labels = []

    test_data = []
    test_labels = []



def imageLoader(imgdim_y: int, filepath_data: str, filepath_lbls: str, arr_data, arr_lbls):
    data_file  = open(filepath_data, "r")

    with open(filepath_lbls, "r") as lbls_file:

        for readline in lbls_file:
            label = [int(x) for x in readline.split()]

            image_array2d = []
            for row in range(imgdim_y):
                image_row = data_file.readline() 
                image_row.pop()
                image_array2d.insert(image_row)

            arr_data.append(AsciImage(image_array2d))

    data_file.close()






def nt_ReadDigitData_Train():
    data_filepath  = os.path.join(A4_PATH, "data/data/digitdata/trainingimages")
    lbls_filepath  = os.path.join(A4_PATH, "data/data/digitdata/traininglabels")

    imageLoader(70, data_filepath, lbls_filepath, mnistDataset.training_data, mnistDataset.training_labels)    

def nt_ReadDigitData_Valid():
    data_filepath  = os.path.join(A4_PATH, "data/data/digitdata/validationimages")
    lbls_filepath  = os.path.join(A4_PATH, "data/data/digitdata/validationlabels")

    imageLoader(70, data_filepath, lbls_filepath, mnistDataset.validation_data, mnistDataset.validation_labels) 

def nt_ReadDigitData_Test():
    data_filepath  = os.path.join(A4_PATH, "data/data/digitdata/testimages")
    lbls_filepath  = os.path.join(A4_PATH, "data/data/digitdata/testlabels")

    imageLoader(70, data_filepath, lbls_filepath, mnistDataset.test_data, mnistDataset.test_labels)






def nt_ReadFaceData_Train():
    data_filepath  = os.path.join(A4_PATH, "data/data/facedata/facedatatrain")
    lbls_filepath  = os.path.join(A4_PATH, "data/data/facedata/facedatatrainlabels")

    imageLoader(data_filepath, lbls_filepath, mnistDataset.training_data, mnistDataset.training_labels)    

def nt_ReadFaceData_Test():
    data_filepath  = os.path.join(A4_PATH, "data/data/facedata/facedatavalidation")
    lbls_filepath  = os.path.join(A4_PATH, "data/data/facedata/facedatavalidationlabels")

    imageLoader(data_filepath, lbls_filepath, mnistDataset.test_data, mnistDataset.test_labels)

def nt_ReadFaceData_Valid():
    data_filepath  = os.path.join(A4_PATH, "data/data/facedata/facedatatest")
    lbls_filepath  = os.path.join(A4_PATH, "data/data/facedata/facedatatestlabels")

    imageLoader(data_filepath, lbls_filepath, mnistDataset.validation_data, mnistDataset.validation_labels) 



def nt_loadall():
    nt_ReadDigitData_Train()
    nt_ReadDigitData_Valid()
    nt_ReadDigitData_Test()

    nt_ReadFaceData_Train()
    nt_ReadFaceData_Valid()
    nt_ReadFaceData_Test()