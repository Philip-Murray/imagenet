import os, sys

A4_PATH = os.path.dirname(os.path.realpath(__file__))

class AsciImage:
    def __init__(self, array2d, label: int):
        self.pixels = array2d
        self.label  = label

    def print(self):
        for img_row in self.pixels:
            print(img_row)



class mnistDataset: #namespace 
    training_data = []
    validation_data = []
    test_data = []

    dim_y = 28
    dim_x = 28

class faceDataset: #namespace
    training_data = []
    validation_data = []
    test_data = []

    dim_x = 60
    dim_y = 70



def imageLoader(imgdim_y: int, filepath_data: str, filepath_lbls: str, arr_data):
    data_file  = open(filepath_data, "r")

    with open(filepath_lbls, "r") as lbls_file:

        for readline in lbls_file:
            label = [int(x) for x in readline.split()]

            image_array2d = []
            for row in range(imgdim_y):
                image_row = list(data_file.readline()) 
                image_row.pop()
                image_array2d.append(image_row)

            arr_data.append(AsciImage(image_array2d, label))

    data_file.close()



#DIGIT DATA --- 
def nt_ReadDigitData_Train():
    data_filepath  = os.path.join(A4_PATH, "data/data/digitdata/trainingimages")
    lbls_filepath  = os.path.join(A4_PATH, "data/data/digitdata/traininglabels")

    imageLoader(mnistDataset.dim_y, data_filepath, lbls_filepath, mnistDataset.training_data)    

def nt_ReadDigitData_Valid():
    data_filepath  = os.path.join(A4_PATH, "data/data/digitdata/validationimages")
    lbls_filepath  = os.path.join(A4_PATH, "data/data/digitdata/validationlabels")

    imageLoader(mnistDataset.dim_y, data_filepath, lbls_filepath, mnistDataset.validation_data) 

def nt_ReadDigitData_Test():
    data_filepath  = os.path.join(A4_PATH, "data/data/digitdata/testimages")
    lbls_filepath  = os.path.join(A4_PATH, "data/data/digitdata/testlabels")

    imageLoader(mnistDataset.dim_y, data_filepath, lbls_filepath, mnistDataset.test_data)


#FACE DATA ---
def nt_ReadFaceData_Train():
    data_filepath  = os.path.join(A4_PATH, "data/data/facedata/facedatatrain")
    lbls_filepath  = os.path.join(A4_PATH, "data/data/facedata/facedatatrainlabels")

    imageLoader(faceDataset.dim_y, data_filepath, lbls_filepath, faceDataset.training_data)    

def nt_ReadFaceData_Test():
    data_filepath  = os.path.join(A4_PATH, "data/data/facedata/facedatavalidation")
    lbls_filepath  = os.path.join(A4_PATH, "data/data/facedata/facedatavalidationlabels")

    imageLoader(faceDataset.dim_y, data_filepath, lbls_filepath, faceDataset.test_data)

def nt_ReadFaceData_Valid():
    data_filepath  = os.path.join(A4_PATH, "data/data/facedata/facedatatest")
    lbls_filepath  = os.path.join(A4_PATH, "data/data/facedata/facedatatestlabels")

    imageLoader(faceDataset.dim_y, data_filepath, lbls_filepath, faceDataset.validation_data) 



def nt_init():
    nt_ReadDigitData_Train()
    nt_ReadDigitData_Valid()
    nt_ReadDigitData_Test()

    nt_ReadFaceData_Train()
    nt_ReadFaceData_Valid()
    nt_ReadFaceData_Test()
