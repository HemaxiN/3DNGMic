import tifffile
import numpy as np
import os

def aux_transpose(image):
    #the output image should be (X,Y,Z)
    original_0 = np.shape(image)[0]
    original_1 = np.shape(image)[1]
    original_2 = np.shape(image)[2]
    index_min = np.argmin([original_0, original_1, original_2])
    if index_min == 0:
        image = image.transpose(1,2,0)
    elif index_min == 1:
        image = image.transpose(0,2,1)
    return image

def convert_npy_tif(dir_, save_dir):
    for name_ in os.listdir(dir_):
        file_ = np.load(os.path.join(dir_, name_))
        file_ = aux_transpose(file_)
        aux_file = np.zeros((np.shape(file_)[0], np.shape(file_)[1], np.shape(file_)[2],3)).astype('uint8')
        aux_file[:,:,:,:2] = file_
        tifffile.imwrite(os.path.join(save_dir, name_.replace('.npy', '.tif')), aux_file)