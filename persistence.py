import os, pickle, loader, features
from algorithms.ann import *
from algorithms.perceptron import *
from algorithms.bayes import *


A4_PATH = os.path.dirname(os.path.realpath(__file__))
SAVE_DIR = os.path.join(A4_PATH, "/savefiles/")
IMGDATA_FILE = os.path.join(A4_PATH, "imagedata.pickle") 



class ModelSession:

    def iterator(self):
        return self.model_array
    def face_iterator(self):
        return self.model_array[0:3]
    def mnist_iterator(self):
        return self.model_array[3:6]

    def get_model(self, dataset_type, model_id):
        return self.model_array[3*dataset_type + model_id]
        
        

def NewSession():
    ns = ModelSession()

    ns.mnist_p = MultiClassPerceptron(features.mnist.featureVectorSize, 10)
    ns.mnist_b = MultiClassNaiveBayes(features.mnist.featureVectorSize, 10)
    ns.mnist_n = MultiClassNeuralNetwork(features.mnist.featureVectorSize, 10) #hardcode 

    ns.face_p = BinaryPerceptron(features.faces.featureVectorSize)
    ns.face_b = BinaryNaiveBayes(features.faces.featureVectorSize)
    ns.face_n = BinaryNeuralNetwork(features.faces.featureVectorSize)
    
    ns.model_array = [ns.face_p, ns.face_b, ns.face_n,   ns.mnist_p, ns.mnist_b, ns.mnist_n]
    return ns





#Serialization for the image files below
#I wrote the image datasets to be static class members, which cannot be serialized automatically
#Therefore, a member-by-member serializaztion method is used for each static member

class ImageDatabase:
    pass

def savedatabase():

    db = ImageDatabase()

    db.lm_train = loader.mnistDataset.training_data
    db.lm_valid = loader.mnistDataset.validation_data
    db.lm_test  = loader.mnistDataset.test_data

    db.lf_train = loader.faceDataset.training_data
    db.lf_valid = loader.faceDataset.validation_data
    db.lf_test  = loader.faceDataset.test_data

    db.lm_dim_x = loader.mnistDataset.dim_x
    db.lm_dim_y = loader.mnistDataset.dim_y
    db.lf_dim_x = loader.faceDataset.dim_x
    db.lf_dim_y = loader.faceDataset.dim_y


    db.fm_X_train = features.mnist.X_train
    db.fm_Y_train = features.mnist.Y_train
    db.fm_X_valid = features.mnist.X_valid
    db.fm_Y_valid = features.mnist.Y_valid
    db.fm_X_test = features.mnist.X_test
    db.fm_Y_test = features.mnist.Y_test

    db.ff_X_train = features.faces.X_train
    db.ff_Y_train = features.faces.Y_train
    db.ff_X_valid = features.faces.X_valid
    db.ff_Y_valid = features.faces.Y_valid
    db.ff_X_test = features.faces.X_test
    db.ff_Y_test = features.faces.Y_test

    db.fm_dim_x = features.mnist.dim_x
    db.fm_dim_y = features.mnist.dim_y
    db.ff_dim_x = features.faces.dim_x
    db.ff_dim_y = features.faces.dim_y
    db.fm_fvl = features.mnist.featureVectorSize
    db.ff_fvl = features.faces.featureVectorSize

    pickle_out = open(IMGDATA_FILE, "wb")
    pickle.dump(db, pickle_out)
    pickle_out.close()
    


def loaddatabase():
    pickle_in = open("imagedata.pickle", "rb")

    db = pickle.load(pickle_in)

    loader.mnistDataset.training_data = db.lm_train
    loader.mnistDataset.validation_data = db.lm_valid
    loader.mnistDataset.test_data = db.lm_test

    loader.faceDataset.training_data = db.lf_train
    loader.faceDataset.validation_data = db.lf_valid
    loader.faceDataset.test_data = db.lf_test

    loader.mnistDataset.dim_x = db.lm_dim_x
    loader.mnistDataset.dim_y = db.lm_dim_y
    loader.faceDataset.dim_x = db.lf_dim_x
    loader.faceDataset.dim_y = db.lf_dim_y


    features.mnist.X_train = db.fm_X_train
    features.mnist.Y_train = db.fm_Y_train 
    features.mnist.X_valid = db.fm_X_valid
    features.mnist.Y_valid = db.fm_Y_valid
    features.mnist.X_test = db.fm_X_test
    features.mnist.Y_test = db.fm_Y_test

    features.faces.X_train = db.ff_X_train
    features.faces.Y_train = db.ff_Y_train
    features.faces.X_valid = db.ff_X_valid
    features.faces.Y_valid = db.ff_Y_valid
    features.faces.X_test = db.ff_X_test
    features.faces.Y_test = db.ff_Y_test

    features.mnist.dim_x = db.fm_dim_x
    features.mnist.dim_y = db.fm_dim_y
    features.faces.dim_x = db.ff_dim_x
    features.faces.dim_y = db.ff_dim_y
    features.mnist.featureVectorSize = db.fm_fvl
    features.faces.featureVectorSize = db.ff_fvl

    pickle_in.close()

