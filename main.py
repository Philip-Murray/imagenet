

import sys, os
import loader, features, persistence, pickle



def saveprogress(x):
    pickle_out = open("dict.pickle","wb")
    pickle.dump(x, pickle_out)
    pickle_out.close()
    

def loadprogress():
    pickle_in = open("dict.pickle", "rb")
    x = pickle.load(pickle_in)
    return x


class A:
    def hi(self):
        print("HI")

    def bye(self):
        print("bye")

class B(A):
    def bye(self):
        print("B")

def t1():
    u = B()
    u.hi()
    u.bye()

    saveprogress(u)

def t2():
    u = loadprogress()
    u.hi()
    u.bye()
    super(type(u), u).bye()

t2()

quit()

A4_PATH = os.path.dirname(os.path.realpath(__file__))

class g_params:
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

#LoadImages()


def LoadSession(force_not_present=False):
    if os.path.exists(os.path.join(persistence.SAVE_DIR, g_params.session_filename)) and (not force_not_present):
        persistence.loadprogress(g_params.session_filename)
    else:
        g_params.session = persistence.ModelSession()

def SaveSession():
    persistence.saveprogress(g_params.session, g_params.session_filename)

import model
model.go()

print(len(sys.argv))

#session = 

#features.faces.printImage(0)
#loader.mnistDataset.test_data[0].print()
#loader.mnistDataset.test_data[0].print()
#loader.faceDataset.test_data[0].print()

quit()

A4_PATH = os.path.dirname(os.path.realpath(__file__))

fn  = os.path.join(A4_PATH, "data/data/facedata/facedatatrain")

f = open(fn, "r")


u = f.readline()

print(len(u))
