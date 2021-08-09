import os, pickle

A4_PATH = os.path.dirname(os.path.realpath(__file__))
SAVE_DIR = os.path.join(A4_PATH, "savefiles")



class SaveData:
    pass


def saveprogress(filename="default"):
    pickle_out = open("dict.pickle","wb")
    pickle.dump(a, pickle_out)
    pickle_out.close()

    pass
    

def loadprogress(filename="default"):
    pickle_in = open("dict.pickle", "rb")
    ex = pickle.load(pickle_in)
    print(ex.b.y)

    pass


