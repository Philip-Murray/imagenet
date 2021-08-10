

import sys, os
import loader, features, persistence

A4_PATH = os.path.dirname(os.path.realpath(__file__))






def LoadImages():
    if os.path.exists(persistence.IMGDATA_FILE):
        persistence.loaddatabase()
    else:
        loader.nt_init()
        features.init()
        persistence.savedatabase()


LoadImages()

features.faces.printImage(0)
loader.mnistDataset.test_data[0].print()


#loader.mnistDataset.test_data[0].print()
#loader.faceDataset.test_data[0].print()

quit()

A4_PATH = os.path.dirname(os.path.realpath(__file__))

fn  = os.path.join(A4_PATH, "data/data/facedata/facedatatrain")

f = open(fn, "r")


u = f.readline()

print(len(u))
