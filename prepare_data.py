import numpy as np
from glob import glob
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2grey
from skimage import exposure

# Function for preparing raw png files
def prepare_raws(x, sizes, three_channels = False):
    if three_channels == False:
        img = imread(x)[:, 150:1058, :]  # Read and crop image
        img = rgb2grey(img)   # to grey image with values from 0 to 1.
        img = exposure.equalize_adapthist(img, clip_limit=0.03)
        img = resize(img, (sizes, sizes), mode="symmetric")  # resizing
        img = np.reshape(img, [sizes, sizes, 1])  # reshaping from [h,w] to [h,w,channel] here we have one channel
        return img

    if three_channels == True:
        img = imread(x)[:, 150:1058, :]  # Read and crop image
        img = resize(img, (sizes, sizes), mode="symmetric")  # resizing
        return img

# function for preparing labels files
def prepare_labels(x, sizes):
    img = imread(x)[:, 150:1058, :]  # Read and crop image
    img = rgb2grey(img)  # to grey image with 1 channel. Need to make mask
    img = resize(img, (sizes, sizes), mode="symmetric")  # resizing
    # img[img == 0] = 1
    img[img < 0] = 1  # binarization to 0 and 1 view
    img = np.expand_dims(img, axis=2)
    return img

def main_thresh_data(three_channels):

    # Read and sorting pathes to data
    names_raw = glob('data/Raws/*/*.png')
    names_labels = glob('data/Labels/*/*.png')
    names_raw.sort()
    names_labels.sort()
    print("Count raw files: ", len(names_raw))
    print("Count labels files: ", len(names_labels))

    # Read and sorting threshold data for pretraining
    names_thr_raw = glob('data/Raws/3/*.png')
    names_thr_labels = glob('data/threshold_masks/*.png')
    names_thr_raw.sort()
    names_thr_labels.sort()
    print("Count raw_th files: ", len(names_thr_raw))
    print("Count labels_th files: ", len(names_thr_labels))

    # variable for resizing
    sizes = 256

    raws_th = np.stack([prepare_raws(i, sizes=sizes, three_channels=three_channels) for i in names_thr_raw], 0)
    labels_th = np.stack([prepare_labels(i, sizes=sizes) for i in names_thr_labels], 0)

    print(raws_th.shape)
    print(labels_th.shape)

    raws = np.stack([prepare_raws(i, sizes=sizes, three_channels=three_channels) for i in names_raw], 0)
    labels = np.stack([prepare_labels(i, sizes=sizes) for i in names_labels], 0)

    print(raws_th.shape)
    print(labels_th.shape)
    print(raws.shape)
    print(labels.shape)

    return raws_th, labels_th, raws, labels

def prepare_data(three_channels=False):

    raws_th, labels_th, raws, labels = main_thresh_data(three_channels=three_channels)
    return raws_th, labels_th, raws, labels






