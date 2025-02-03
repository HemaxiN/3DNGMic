import numpy as np
import os
from keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
from numpy import zeros
import pickle
import cv2
import math
import random

def augm(image, vx, vy, vz):
    
    image = image/255.0
    image_aux = image.copy()
    image_aux = np.zeros(np.shape(image_aux))
    
    if random.uniform(0,1)<0.5:
        image, vx, vy, vz = vertical_flip(image, vx, vy, vz)
    if random.uniform(0,1)<0.5:
        image, vx, vy, vz = horizontal_flip(image, vx, vy, vz)
    if random.uniform(0,1)<0.5:
        angle = np.random.choice(np.arange(0,360,90))
        image, vx, vy, vz = rotation(image, vx, vy, vz, angle)
        
    if np.shape(image)[0] % 2 != 0:
        bbox_aux = np.zeros((np.shape(image)[0]+1, np.shape(image)[1], np.shape(image)[2]))
        bbox_aux[0:np.shape(image)[0],0:np.shape(image)[1],0:np.shape(image)[2]] = image
        image = bbox_aux.copy()
        del bbox_aux
    if np.shape(image)[1] % 2 != 0:
        bbox_aux = np.zeros((np.shape(image)[0], np.shape(image)[1]+1, np.shape(image)[2]))
        bbox_aux[0:np.shape(image)[0],0:np.shape(image)[1],0:np.shape(image)[2]] = image
        image = bbox_aux.copy()
        del bbox_aux
    if np.shape(image)[2] % 2 != 0:
        bbox_aux = np.zeros((np.shape(image)[0], np.shape(image)[1], np.shape(image)[2]+1))
        bbox_aux[0:np.shape(image)[0],0:np.shape(image)[1],0:np.shape(image)[2]] = image
        image = bbox_aux.copy()
        del bbox_aux
        
    
    vy = image.shape[1]/2
    vx = image.shape[0]/2
    vz = image.shape[2]/2
    
    #print('size image {}'.format(image.shape))
    #print('vx {}'.format(vx))
    #print('vy {}'.format(vy))
    
    image = (image*255.0).astype('uint8')
    
    return image, vx, vy, vz

##auxiliary function to rotate the point
def rotate_around_point_lowperf(image, pointx, pointy, angle):
    """Rotate a point around a given point.
    
    I call this the "low performance" version since it's recalculating
    the same values more than once [cos(radians), sin(radians), x-ox, y-oy).
    It's more readable than the next function, though.
    """
    radians = (np.pi*angle)/(180)
    x, y = pointx, pointy
    ox, oy = image.shape[0]/2, image.shape[1]/2
    qx = ox + math.cos(radians) * (x - ox) + math.sin(radians) * (y - oy)
    qy = oy - math.sin(radians) * (x - ox) + math.cos(radians) * (y - oy)
    return qx, qy

def rotation(image, vx, vy, vz, angle):
    h, w, d = image.shape  # Get original dimensions
    center = (w / 2, h / 2)

    # Compute the new bounding box size after rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    
    # Adjust transformation matrix to consider new center
    new_center = (new_w / 2, new_h / 2)
    M[0, 2] += new_center[0] - center[0]
    M[1, 2] += new_center[1] - center[1]

    # Rotate each slice in the z-dimension
    rot_image = np.zeros((new_h, new_w, d), dtype=image.dtype)
    for z in range(d):
        rot_image[:, :, z] = cv2.warpAffine(image[:, :, z], M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    rvx, rvy = rotate_around_point_lowperf(image, vx, vy, angle)

    return rot_image, rvx, rvy, vz

##vertical flip
def vertical_flip(image, vx, vy, vz):
    flippedimage = np.zeros(np.shape(image))
    for z in range(0, flippedimage.shape[2]):
        flippedimage[:,:,z] = cv2.flip(image[:,:,z], 0)
    fvx, fvy, fvz = vx, (image.shape[1]-vy-1), vz
    return flippedimage, fvx, fvy, fvz

##horizontal flip
def horizontal_flip(image, vx, vy, vz):
    flippedimage = np.zeros(np.shape(image))
    for z in range(0, flippedimage.shape[2]):
        flippedimage[:,:,z] = cv2.flip(image[:,:,z], 1)
    fvx, fvy, fvz = image.shape[0]-vx-1, vy, vz
    return flippedimage, fvx, fvy, fvz

def inputs_generation(save_dir_masks, save_dir_vecs, nb_patches):

    patch_size_x_y = 128 #patch size along x and y directions
    img_sizex = 128
    img_sizey = 128
    img_sizez = 64
    patch_size_z = 64 #patch size along z direction

    nb_pairs_ppatch_max = 19 #max nb of nucleus-Golgi pairs per patch
    nb_pairs_ppatch_min = 11 #min nb of nucleus-Golgi pairs per patch

    with open('object_shapes/nuclei_shapes.pkl', 'rb') as file:
        n_patches_dict = pickle.load(file)

    with open('object_shapes/Golgi_shapes.pkl', 'rb') as file:
        g_patches_dict = pickle.load(file)

    with open('object_shapes/Golgi_centroids.pkl', 'rb') as file:
        g_centroids = pickle.load(file)

    with open('object_shapes/nuclei_centroids.pkl', 'rb') as file:
        n_centroids = pickle.load(file)

    for i in range(0, nb_patches):
        patch = np.zeros((patch_size_x_y, patch_size_x_y, patch_size_z, 3)) #initialize
        #ith patch
        patch_instance_nuclei = np.zeros((patch_size_x_y, patch_size_x_y, patch_size_z))
        patch_instance_golgi = np.zeros((patch_size_x_y, patch_size_x_y, patch_size_z))
        nb_pairs = np.random.choice(np.arange(nb_pairs_ppatch_min, nb_pairs_ppatch_max)) #nb of 
        #nucleus-Golgi pairs in the ith patch

        vectors = [] # np array, (size Nx6) where N is the number of vectors in the 
        #patch, and (vx,vy,vz) are the components of the nucleus-Golgi vector

        k_nuclei = 1
        k_golgi = 1

        for j in range(0, nb_pairs):
            accept = False
            while not accept:
                nctrx = np.random.choice(np.arange(10,patch_size_x_y-10))
                nctry = np.random.choice(np.arange(10,patch_size_x_y-10))
                nctrz = np.random.choice(np.arange(20,40))

                ngdistx = np.random.choice(np.arange(-10,10))
                ngdisty = np.random.choice(np.arange(-10,10))
                ngdistz = np.random.choice(np.arange(-4,4))

                if np.sign(ngdistx)==-1:
                    gctrx = max(nctrx+ngdistx, 0)
                else:
                    gctrx = min(nctrx+ngdistx, patch_size_x_y)
                if np.sign(ngdisty)==-1:
                    gctry = max(nctry+ngdisty, 0)
                else:
                    gctry = min(nctry+ngdisty, patch_size_x_y)
                if np.sign(ngdistz)==-1:
                    gctrz = max(nctrz+ngdistz, 0)
                else:
                    gctrz = min(nctrz+ngdistz, patch_size_z)

                volxy = 10
                volz = 5

                limnxi = max(nctrx-volxy,0)
                limnxs = min(nctrx+volxy, patch_size_x_y)

                limnyi = max(nctry-volxy,0)
                limnys = min(nctry+volxy, patch_size_x_y)

                limnzi = max(nctrz-volz,0)
                limnzs = min(nctrz+volz, patch_size_z)    

                limgxi = max(gctrx-volxy,0)
                limgxs = min(gctrx+volxy, patch_size_x_y)

                limgyi = max(gctry-volxy,0)
                limgys = min(gctry+volxy, patch_size_x_y)

                limgzi = max(gctrz-volz,0)
                limgzs = min(gctrz+volz, patch_size_z)

                if np.sum(patch[limnxi:limnxs, limnyi:limnys, limnzi:limnzs, 1]) == 0 and np.sum(patch[limgxi:limgxs, limgyi:limgys, limgzi:limgzs, 0]) == 0:

                    nrandom = np.random.choice(np.arange(0,len(n_patches_dict)))
                    grandom = np.random.choice(np.arange(0,len(g_patches_dict)))

                    n_selected_patch = n_patches_dict[nrandom]
                    g_selected_patch = g_patches_dict[grandom]

                    while np.count_nonzero(n_selected_patch)<500:  #ignore small nuclei
                        nrandom = np.random.choice(np.arange(0,len(n_patches_dict)))
                        n_selected_patch = n_patches_dict[nrandom]

                    while np.count_nonzero(g_selected_patch)<=100:  #ignore small golgi
                        grandom = np.random.choice(np.arange(0,len(g_patches_dict)))
                        g_selected_patch = g_patches_dict[grandom]


                    npctrx = n_centroids[nrandom][0]
                    npctry = n_centroids[nrandom][1]
                    npctrz = n_centroids[nrandom][2]

                    gpctrx = g_centroids[grandom][0]
                    gpctry = g_centroids[grandom][1]
                    gpctrz = g_centroids[grandom][2]

                    n_selected_patch, npctrx, npctry, npctrz = augm(n_selected_patch, npctrx, npctry, npctrz)
                    g_selected_patch, gpctrx, gpctry, gpctrz = augm(g_selected_patch, gpctrx, gpctry, gpctrz)
                    #print('n selected {}'.format(n_selected_patch.shape))
                    #print('g selected {}'.format(g_selected_patch.shape))

                    nsize_x = np.shape(n_selected_patch)[0]
                    nsize_y = np.shape(n_selected_patch)[1]
                    nsize_z = np.shape(n_selected_patch)[2]

                    gsize_x = np.shape(g_selected_patch)[0]
                    gsize_y = np.shape(g_selected_patch)[1]
                    gsize_z = np.shape(g_selected_patch)[2]


                    nxmin = max(0,int(nctrx-(nsize_x/2)))
                    nxmax = min(patch_size_x_y,int(nctrx+(nsize_x/2)))
                    nymin = max(0,int(nctry-(nsize_y/2)))
                    nymax = min(patch_size_x_y,int(nctry+(nsize_y/2)))
                    nzmin = max(0,int(nctrz-(nsize_z/2)))
                    nzmax = min(patch_size_x_y,int(nctrz+(nsize_z/2)))

                    npxmin = max(0,int(npctrx-(nsize_x/2)))
                    npxmax = min(nsize_x,int(npctrx+(nsize_x/2)))
                    npymin = max(0,int(npctry-(nsize_y/2)))
                    npymax = min(nsize_y,int(npctry+(nsize_y/2)))
                    npzmin = max(0,int(npctrz-(nsize_z/2)))
                    npzmax = min(nsize_z,int(npctrz+(nsize_z/2)))


                    gxmin = max(0,int(gctrx-(gsize_x/2)))
                    gxmax = min(patch_size_x_y,int(gctrx+(gsize_x/2)))
                    gymin = max(0,int(gctry-(gsize_y/2)))
                    gymax = min(patch_size_x_y,int(gctry+(gsize_y/2)))
                    gzmin = max(0,int(gctrz-(gsize_z/2)))
                    gzmax = min(patch_size_x_y,int(gctrz+(gsize_z/2)))

                    gpxmin = max(0,int(gpctrx-(gsize_x/2)))
                    gpxmax = min(gsize_x,int(gpctrx+(gsize_x/2)))
                    gpymin = max(0,int(gpctry-(gsize_y/2)))
                    gpymax = min(gsize_y,int(gpctry+(gsize_y/2)))
                    gpzmin = max(0,int(gpctrz-(gsize_z/2)))
                    gpzmax = min(gsize_z,int(gpctrz+(gsize_z/2)))


                    if nxmin!=0 and nxmax!=128 and nymin!=0 and nymax!=128 and nzmin!=0 and nzmax!=128 and gxmin!=0 and gxmax!=128 and gymin!=0 and gymax!=128 and gzmin!=0 and gzmax!=128:
                        accept = True
                        random_patch = np.zeros((patch_size_x_y,patch_size_x_y,patch_size_z,3))

                        n_selected_patch = n_selected_patch.astype('uint8')
                        #kernel = np.ones((5,5),np.uint8)
                        #n_selected_patch = cv2.erode(n_selected_patch, kernel, iterations=1)

                        random_patch[nxmin:nxmax, nymin:nymax, nzmin:nzmax, 1] = n_selected_patch[npxmin:npxmax, npymin:npymax, npzmin:npzmax]
                        patch[:,:,:,1]=np.logical_or(patch[:,:,:,1], random_patch[:,:,:,1])
                        random_patch[gxmin:gxmax, gymin:gymax, gzmin:gzmax, 0] = g_selected_patch[gpxmin:gpxmax, gpymin:gpymax, gpzmin:gpzmax]
                        patch[:,:,:,0]=np.logical_or(patch[:,:,:,0], random_patch[:,:,:,0])
                        vectors.append([nctrx,nctry,nctrz,gctrx,gctry,gctrz])

                        patch_instance_nuclei[random_patch[:,:,:,1]!=0] = k_nuclei
                        patch_instance_golgi[random_patch[:,:,:,0]!=0] = k_golgi

                        k_nuclei += 1
                        k_golgi += 1


        patch = patch*255.0
        patch = patch.transpose(2,0,1,3)
        patch = patch.astype('uint8')
        np.save(os.path.join(save_dir_masks, str(i) + '.npy'), patch.transpose(1,2,0,3)[:,:,:,0:2])

        patch_instance_nuclei = patch_instance_nuclei.astype('uint8')
        patch_instance_golgi = patch_instance_golgi.astype('uint8')
        #tifffile.imwrite(os.path.join(save_dir_masks_instance_n, str(i+500) + '.tif'), patch_instance_nuclei)
        #tifffile.imwrite(os.path.join(save_dir_masks_instance_g, str(i+500) + '.tif'), patch_instance_golgi)

        np.save(os.path.join(save_dir_vecs, str(i) + '.npy'), vectors)



def load_real_samples(input_path, ix):
    X2=zeros((len(ix),64,128,128,2),dtype='float32')
    k=0
    for i in ix:
        mask = np.load(os.path.join(input_path , str(i)+'.npy'))
        mask = mask.transpose(2,0,1,3) 
        X2[k,:]=(mask-127.5) /127.5
        k=k+1
    return X2

# generate samples and save as a plot and save the model
def summarize_performance(g_model, samples_, input_path, save_dir):
    for s in samples_:
        # select a sample of input images
        X_realA = load_real_samples(input_path, [s])
        # generate a batch of fake samples
        X_fakeB = g_model.predict(X_realA)
        # scale all pixels from [-1,1] to [0,1]
        X_fakeB = (X_fakeB + 1) / 2.0
        filename = os.path.join(save_dir, str(s) + '.npy')
        save_img = np.zeros((64,128,128,2)).astype('uint8')
        save_img[:,:,:,0:2] = X_fakeB[0]*255
        save_img = save_img.transpose(1,2,0,3)
        save_img = save_img.astype('uint8')
        np.save(filename, save_img)
        #print(str(s))

        
def predict(models_path, input_path, save_dir, patch_numbers):
      g_model = load_model(models_path)
      summarize_performance(g_model, patch_numbers, input_path, save_dir)

