import numpy as np
import os
import pandas as pd
from skimage.metrics import structural_similarity as ssim

def compute_SSIM(img1_dir, img2_dir):
    performance_metrics_SSIM = pd.DataFrame(columns = ["Image", "SSIM"])
    for img1, img2 in zip(sorted(os.listdir(img1_dir)), sorted(os.listdir(img2_dir))):
        images1 = np.load(os.path.join(img1_dir, img1))
        images1 = images1.transpose(2,0,1,3)
        images2 = np.load(os.path.join(img2_dir, img2))
        
        images2_aux = np.zeros((64,128,128,3)).astype('uint8')
        images2_aux[:,:,:,0:2] = images2
        images2 = images2_aux
        
        images1_aux = np.zeros((64,128,128,3)).astype('uint8')
        images1_aux[:,:,:,0:2] = images1
        images1 = images1_aux

        ssim_value = ssim(images1, images2, data_range=images2.max() - images2.min(),
                          channel_axis=3, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, K1=0.01, K2=0.03)

        res = {"Image": img1, "SSIM": ssim_value}

        row = len(performance_metrics_SSIM)
        performance_metrics_SSIM.loc[row] = res

    performance_metrics_SSIM.to_csv('ssim_values.csv', index=False)
    print('mean SSIM {}'.format(performance_metrics_SSIM.agg({'SSIM': np.mean})))