import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,20)
import numpy as np
import os

def normalization(mip_img):
    minval = np.percentile(mip_img, 0.05)
    maxval = np.percentile(mip_img, 99.95)
    mip_img = np.clip(mip_img, minval, maxval)
    mip_img = (((mip_img - minval) / (maxval - minval)) * 255).astype('uint8')
    return mip_img

def visualize_dataset_2d(image_dir, mask_dir, heatmap_dir, vector_dir):
    for img in os.listdir(image_dir):
        image = np.load(os.path.join(image_dir, img))
        heatmap = np.load(os.path.join(heatmap_dir, img))
        mask = np.load(os.path.join(mask_dir, img))
        vectors = np.load(os.path.join(vector_dir, img))

        image_aux = np.zeros((128,128,64,3)).astype('uint8')
        image_aux[:,:,:,:2] = image
        
        image_aux[:,:,:,0] = normalization(image_aux[:,:,:,0])
        image_aux[:,:,:,1] = normalization(image_aux[:,:,:,1])
        image_aux[:,:,:,2] = normalization(image_aux[:,:,:,2])

        mask_aux = np.zeros((128,128,64,3)).astype('uint8')
        mask_aux[:,:,:,:2] = mask
        
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)

        ax1.imshow(np.max(image_aux, axis=-2))
        ax1.set_title('Image Projection and Nucleus-GolgiVectors')
        ax1.set_axis_off()
        
        for vec in vectors:
            ax1.arrow(vec[1], vec[0], vec[4] - vec[1], vec[3] - vec[0], color='w', width=0.5)
        
        ax2.imshow(np.max(heatmap[:,:,:,0], axis=-1), cmap='jet')
        ax2.set_title('Golgi Gaussian Heatmap Projection')
        ax2.set_axis_off()
        ax3.imshow(np.max(heatmap[:,:,:,1], axis=-1), cmap='jet')
        ax3.set_title('Nuclei Gaussian Heatmap Projection')
        ax3.set_axis_off()
        ax4.imshow(np.max(mask_aux, axis=-2))
        ax4.set_title('Mask Projection')
        ax4.set_axis_off()