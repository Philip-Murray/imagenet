import os, sys, pickle

A4_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))





class B:
    def __init__(self):
        self.y = 5


class A:
    def __init__(self):
        self.x = 3
        self.b = B()



def f1():
    a = A()
    pickle_out = open("dict.pickle","wb")
    pickle.dump(a, pickle_out)
    pickle_out.close()

def f2():
    pickle_in = open("dict.pickle", "rb")
    ex = pickle.load(pickle_in)
    print(ex.b.y)


f = open("./data/data/digitdata/testimages")

class Image:
    def __init__(self, array2d, label):
        self.pixels = array2d
        self.label  = label
        


class MNIST_Dataset:
    training_data = []
    training_labels = []

    test_data = []
    test_labels = []


def MNIST_DataReader(filepath_data, filepath_lbls, arr_data, arr_lbls):

    data_file  = open(filepath_data, "r")
    
    with open(filepath_lbls, "r") as lbls_file:

        for readline in lbls_file:
            label = [int(x) for x in readline.split()]

            image_array2d = []
            for row in range(28):
                image_row = data_file.readline()
                image_array2d.insert(image_row)

            arr_data.append(Image(image_array2d))
            arr_lbls.append()





def nt_Read_DigitData_Test(file_data, file_labels):
    data_file  = open(os.path.join(A4_PATH, "data/data/digitdata/testimages"), "r")
    lbls_file  = open(os.path.join(A4_PATH, "data/data/digitdata/testlabels"), "r")

    while True:
        label = [int(x) for x in lbls_file.readline().split()]

        image_array2d = []

        for row in range(28):
            image_row = data_file.readline()
            image_array2d.insert(image_row)

        

    

    


for i in range(10):
    u = f.readline()
    print(u[23])
    print(len(u))
    #print([x for x in f.readline()])
    #u = v.insert([int(x) for x in f.readline().split()])

