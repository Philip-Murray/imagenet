

import sys, os


import loader, features

loader.nt_loadall()

features.init()


#loader.mnistDataset.test_data[0].print()
#loader.faceDataset.test_data[0].print()

quit()

A4_PATH = os.path.dirname(os.path.realpath(__file__))

fn  = os.path.join(A4_PATH, "data/data/facedata/facedatatrain")

f = open(fn, "r")


u = f.readline()

print(len(u))
