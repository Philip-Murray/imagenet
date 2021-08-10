import os, pickle, loader, features

A4_PATH = os.path.dirname(os.path.realpath(__file__))
SAVE_DIR = os.path.join(A4_PATH, "savefiles")
IMGDATA_FILE = os.path.join(A4_PATH, "imagedata.pickle") 


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



class ImageDatabase:
    pass

def savedatabase():

    db = ImageDatabase()

    ImageDatabase.mnistDataset  = loader.mnistDataset
    ImageDatabase.mnist = features.mnist

    ImageDatabase.faceDataset   = loader.faceDataset
    ImageDatabase.faces = features.faces




    pickle_out = open(IMGDATA_FILE, "wb")
    pickle.dump(db, pickle_out)
    pickle_out.close()
    


def loaddatabase():
    pickle_in = open("imagedata.pickle", "rb")

    db = pickle.load(pickle_in)
    
    loader.mnistDataset = db.mnistDataset
    loader.faceDataset  = db.faceDataset
    features.mnist = db.mnist
    features.faces = db.faces

    pickle_in.close()

