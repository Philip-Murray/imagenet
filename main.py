import pandas

import sys, os, pickle
import loader, features, persistence, model
from algorithms.ann import *
from algorithms.perceptron import *
from algorithms.bayes import *
from experiments import ExperimentContainer, TrainGlobalSessionModels
import experiments




A4_PATH = os.path.dirname(os.path.realpath(__file__))

class g_params:
    loader_ran = False
    session_filename = "default"
    session = None


def LoadImages(force_not_present=False, save_copy_if_not_present=True):
    if os.path.exists(persistence.IMGDATA_FILE) and (not force_not_present):
        persistence.loaddatabase()
    else:
        loader.nt_init()
        features.init()
        if save_copy_if_not_present:
            persistence.savedatabase()




def main_loader():
    if g_params.loader_ran:
        return
    g_params.loader_ran = True
    LoadImages()

def Problem3Experiment(epochs=1):
    main_loader()
    return ExperimentContainer().run(epochs)

main_loader()



if len(sys.argv) > 1:
    arg2 = sys.argv[1]

    if sys.argv[1] == "images":
        if sys.argv[2] == "mnist":
            if sys.argv[3] == "train":
                loader.mnistDataset.training_data[int(sys.argv[4])].print()
                print("Label: "+str(loader.mnistDataset.training_data[int(sys.argv[4])].label))
            else:
                loader.mnistDataset.test_data[int(sys.argv[4])].print()
                print("Label: "+str(loader.mnistDataset.test_data[int(sys.argv[4])].label))

        if sys.argv[2] == "faces":
            if sys.argv[3] == "train":
                loader.faceDataset.training_data[int(sys.argv[4])].print()
                print("Label: "+str(loader.faceDataset.training_data[int(sys.argv[4])].label))
            else:
                loader.faceDataset.test_data[int(sys.argv[4])].print()
                print("Label: "+str(loader.faceDataset.test_data[int(sys.argv[4])].label))

    if sys.argv[1] == "features":
        if sys.argv[2] == "mnist":
            if sys.argv[3] == "train":
                features.mnist.printImage(int(sys.argv[4]), "train")
            else:
                features.mnist.printImage(int(sys.argv[4]), "test")

        if sys.argv[2] == "faces":
            if sys.argv[3] == "train":
                features.faces.printImage(int(sys.argv[4]), "train")
            else:
                features.faces.printImage(int(sys.argv[4]), "test")

    if sys.argv[1] == "session":
        try:
            percent = int(sys.argv[1])
        except:
            percent = 100
        g_params.session = persistence.NewSession()
        print("Training models on "+str(percent)+chr(37)+" of datasets.")

        tg = TrainGlobalSessionModels(percent, g_params.session)
        tg.run(report_progress=True)
        #SaveSession()

        for line in sys.stdin:
            j = line.split(" ")
            if j[0] == "quit" or j[0] == "quit\n":
                quit()

            first = int(j[2])
            last = first + 1
            if len(j) == 4:
                last = int(j[3]) + 1
            
            datasetv = 0
            if j[0] == "mnist":
                datasetv = 1
            
            modeltype = 0
            if j[1] == "nb":
                modeltype = 1
            if j[1] == "ann" or j[1] == "nn":
                modeltype = 2

            experiments.TestGlobalModel(g_params.session, datasetv, modeltype, first, last)

    if sys.argv[1] == "experiment":
        Problem3Experiment()
else:
    print("Please issue a command. Commands may be found in latex pdf.")