import pandas

import sys, os, pickle
import loader, features, persistence, model
from algorithms.ann import *
from algorithms.perceptron import *
from algorithms.bayes import *
from experiments import ExperimentContainer




A4_PATH = os.path.dirname(os.path.realpath(__file__))

class g_params:
    session_filename = "default"
    session = None

class Metrics:
    pass





def LoadImages(force_not_present=False, save_copy_if_not_present=True):
    if os.path.exists(persistence.IMGDATA_FILE) and (not force_not_present):
        persistence.loaddatabase()
    else:
        loader.nt_init()
        features.init()
        if save_copy_if_not_present:
            persistence.savedatabase()

def LoadSession(force_not_present=False):
    if os.path.exists(os.path.join(persistence.SAVE_DIR, g_params.session_filename)) and (not force_not_present):
        persistence.loadprogress(g_params.session_filename)
    else:
        g_params.session = persistence.NewSession()

def SaveSession():
    persistence.saveprogress(g_params.session, g_params.session_filename)

def SetSaveFile(fname: str):
    g_params.session_filename = fname


def Problem3Experiment():
    return ExperimentContainer().run()



def Testing1(session: persistence.ModelSession, epochs_per_10p=3):
    for i in range(1, 11):
        percent = (i*10)
        print("Training models to "+str(percent)+" percent of data:")

        frac = percent / 100


        mnist_training_length = features.mnist.X_train.shape[0]
        faces_training_length = features.faces.X_train.shape[0]

        mX_train = features.mnist.X_train[0:int(frac * mnist_training_length)]
        mY_train = features.mnist.Y_train[0:int(frac * mnist_training_length)]

        fX_train = features.faces.X_train[0:int(frac * faces_training_length)]
        fY_train = features.faces.Y_train[0:int(frac * faces_training_length)]


        session.mnist_p.fit(mX_train, mY_train, epochs_per_10p)
        session.mnist_b.fit(mX_train, mY_train, epochs_per_10p)
        session.mnist_n.fit(mX_train, mY_train, 3)

        session.face_p.fit(fX_train, fY_train, epochs_per_10p)
        session.face_b.fit(fX_train, fY_train, epochs_per_10p)
        session.face_n.fit(fX_train, fY_train, 3)
        print()
        print()

    print()
    print("Accuracy testing")

    session.mnist_p.accuracy_test(features.mnist.X_test, features.mnist.Y_test, print_ans=True)
    session.mnist_b.accuracy_test(features.mnist.X_test, features.mnist.Y_test, print_ans=True)
    session.mnist_n.accuracy_test(features.mnist.X_test, features.mnist.Y_test, print_ans=True)

    session.face_p.accuracy_test(features.faces.X_test, features.faces.Y_test, print_ans=True)
    session.face_b.accuracy_test(features.faces.X_test, features.faces.Y_test, print_ans=True)
    session.face_n.accuracy_test(features.faces.X_test, features.faces.Y_test, print_ans=True)




def main():
    LoadImages()
    LoadSession()

    #Testing1(g_params.session, 1)

main()
Problem3Experiment()

quit()
#session = 

#features.faces.printImage(0)
#loader.mnistDataset.test_data[0].print()
#loader.mnistDataset.test_data[0].print()
#loader.faceDataset.test_data[0].print()


A4_PATH = os.path.dirname(os.path.realpath(__file__))

fn  = os.path.join(A4_PATH, "data/data/facedata/facedatatrain")

f = open(fn, "r")


u = f.readline()

print(len(u))
