from sys import argv
import os
from PIL import Image
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from process_data import BB_ts_to_curve as bbts
from scipy.misc import imsave
from os.path import join


def feat_map(file, verbose = False):

    with open(file) as f:
        st = f.read()
    json_acceptable_string = st.replace("'", "\"")
    dic = json.loads(json_acceptable_string)

    width = dic['width']
    height = dic['height']
    scene_number = dic['scene']
    init_width = dic['initial_width']
    direc = dic['directory']



    f = open(join("output", "{}_feat.xml").format(direc), "w")
    ct = 0
    f.write(
    """<?xml version="1.0"?>
    <opencv_storage>
    """)
    def cv2np(arr):
        return np.array(arr)/255.0
    def write_feature(arr, ct):
        arr -= 1
        vals = " ".join(map(str, list(arr.flatten())))
        f.write(
        """<feature_{0} type_id="opencv-matrix">
        <rows>{2}</rows>
        <cols>{3}</cols>
        <dt>f</dt>
        <data>
        {1}
        </data></feature_{0}>
    """.format(ct, vals, height, width))




    f_count = 0
    for file in os.listdir(direc):
        if file == ".DS_Store":
            continue


        im = Image.open(join(direc, file))
        p = np.flipud(255-np.array(im))
        print p.shape
        if len(p.shape) > 2:
            p = p[:,:,0]
        distances = np.zeros_like(p[:, :]) * 0.0
        kernel = np.ones((3,3),np.uint8)
        cvimg = cv2.cvtColor(np.stack((p, p, p), axis=2), cv2.COLOR_BGR2GRAY)
        ct = 1
        dilate1 = cvimg
        while 1:
            if np.sum(cv2np(cvimg)) == 0:
                break
            dilation = cv2.dilate(cvimg, kernel, iterations = ct)
            mask = (distances == 0)
            distances += (cv2np(dilation-dilate1) > 0) * mask * ct
            if np.sum(cv2np(dilation)) == len(cv2np(dilation).flatten()):
                break
            dilate1 = dilation
            ct += 1
        write_feature(p/255.0, f_count)
        plt.show()
        print "{}: {}, no blur".format(f_count, file)
        f_count += 1
        sigma = [1, 3, 5]
        for i in sigma:
            arr = 1/(1 + np.exp(distances / i))
            print arr.shape
            write_feature(arr, f_count)
            print "{}, {}, sigma={}".format(f_count, file, i)
            f_count += 1
            if verbose:
                plt.imshow(arr, cmap="viridis")
                plt.show()



    f.write("</opencv_storage>")

if __name__ == "__main__":
    file = argv[1]
    verbose = True
    if len(argv) > 2:
        verbose = False
    feat_map(file, verbose)
