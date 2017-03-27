import os
from shutil import copy as cp

names = ['coupa', 'bookstore', 'gates', 'death']

def mkdir(fname):
    try:
        os.mkdir(fname)
    except:
        pass

for name in names:
    print name
    for i in range(5):
        print i
        path_name = "{}/{}/".format(name, i)
        mkdir(path_name + "walk_traj")
        for file in os.listdir(path_name):
            if file == "walk_basenames.txt" or \
               file == ".DS_Store" or \
               file == "walk_traj":
                continue
            try:
                os.rename(path_name + file, path_name + "walk_traj/" + file)
            except:
                pass
        mkdir(path_name + "walk_imag")
        mkdir(path_name + "walk_feat")
        mkdir(path_name + "walk_output")
        fname = name + "_topdown.jpg"
        cp(fname, path_name + "walk_imag/topdown.jpg")
        fname = name + "_feat.xml"
        cp(fname, path_name + "walk_feat/feature_map.xml")

