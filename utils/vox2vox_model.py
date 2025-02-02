from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
import numpy as np
#from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv3D
from keras.layers import Conv3DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot
import tifffile
import os
import tensorflow as tf
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
from skimage.metrics import structural_similarity as ssim
import pandas as pd

# define the discriminator model
def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_src_image = Input(shape=image_shape)
    # target image input
    in_target_image = Input(shape=image_shape)
    # concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])
    # C64
    d = Conv3D(32, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv3D(64, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv3D(128, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv3D(256, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv3D(256, (4,4,4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv3D(1, (4,4,4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    # compile model
    #opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model

# define an encoder block
def define_encoder_block(layer_in, n_filters, strides_=(1,2,2), batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv3D(n_filters, (4,4,4), strides=strides_, padding='same', kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, strides_=(1,2,2), dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv3DTranspose(n_filters, (4,4,4), strides=strides_, padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g

# define the standalone generator model
def define_generator(image_shape=(64,128,128,2)):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model
    e1 = define_encoder_block(in_image, 32, strides_=(1,2,2), batchnorm=False)
    e2 = define_encoder_block(e1, 64, strides_=(1,2,2))
    e3 = define_encoder_block(e2, 128, strides_=(1,2,2))
    e4 = define_encoder_block(e3, 256, strides_=(2,2,2))
    e5 = define_encoder_block(e4, 256, strides_=(2,2,2))
    e6 = define_encoder_block(e5, 256, strides_=(2,2,2))
    e7 = define_encoder_block(e6, 256, strides_=(2,2,2))
    # bottleneck, no batch norm and relu
    b = Conv3D(512, (4,4,4), strides=(1,1,1), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)
    # decoder model
    d1 = decoder_block(b, e7, 256, strides_=(1,1,1))
    d2 = decoder_block(d1, e6, 256, strides_=(2,2,2))
    d3 = decoder_block(d2, e5, 256, strides_=(2,2,2))
    d4 = decoder_block(d3, e4, 256, strides_=(2,2,2), dropout=False)
    d5 = decoder_block(d4, e3, 128, strides_=(2,2,2), dropout=False)
    d6 = decoder_block(d5, e2, 64, strides_=(1,2,2), dropout=False)
    d7 = decoder_block(d6, e1, 32, strides_=(1,2,2), dropout=False)
    # output
    g = Conv3DTranspose(2, (4,4,4), strides=(1,2,2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
            
    #opt = Adam(lr=0.0002, beta_1=0.5)
    d_model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])            
    # define the source image
    in_src = Input(shape=image_shape)
    # connect the source image to the generator input
    gen_out = g_model(in_src)
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and classification output
    model = Model(in_src, [dis_out, gen_out])
    # compile model
    #opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
    return model
    

def load_real_samples(ix, dir_, img_shape):
    X1=zeros((len(ix),img_shape[0],img_shape[1],img_shape[2],img_shape[3]),dtype='float32')
    X2=zeros((len(ix),img_shape[0],img_shape[1],img_shape[2],img_shape[3]),dtype='float32')
    k=0
    for i in ix:
        image = np.load(os.path.join(dir_, 'outputs/'+str(i)+'.npy')) # RGB image
        image = image.transpose(2,0,1,3)
        mask = np.load(os.path.join(dir_, 'inputs/'+str(i)+'.npy')) # RGB image
        mask = mask.transpose(2,0,1,3)
        X1[k,:]=(image-127.5) /127.5
        X2[k,:]=(mask-127.5) /127.5
        k=k+1
    return [X1, X2]


# select a batch of random samples, returns images and target
def generate_real_samples(n_patches, dir_, img_shape, n_samples, patch_shape):
    # choose random instances
    ix = randint(0, n_patches, n_samples)
    # retrieve selected images
    X1, X2 = load_real_samples(ix, dir_, img_shape)
    #X1, X2 = trainA[ix], trainB[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, 4, patch_shape, patch_shape, 1))
    return [X1, X2], y

# select a batch of random samples, returns images and target
def generate_real_samples_performance(dir_, img_shape, ix, patch_shape):
    # retrieve selected images
    X1, X2 = load_real_samples(ix, dir_, img_shape)
    #X1, X2 = trainA[ix], trainB[ix]
    # generate 'real' class labels (1)
    return [X1, X2]

# select a batch of random samples, returns images and target
def generate_real_samples2(ix, n_samples, patch_shape, dir_, img_shape):
    # retrieve selected images
    X1, X2 = load_real_samples(ix, dir_, img_shape)
    # generate 'real' class labels (1)
    y = ones((n_samples, 4, patch_shape, patch_shape, 1))
    return [X1, X2], y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((len(X), 4, patch_shape, patch_shape, 1))
    return X, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, n_patches_val, val_dir, img_shape, n_samples=5):
    ssim_aux = []
    for ix in range(0,18):
        [XrealB, X_realA] = generate_real_samples_performance(val_dir, img_shape, [ix], 1)
        X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
        ssim_aux.append(ssim(XrealB[0], X_fakeB[0], data_range=X_fakeB.max() - X_fakeB.min(),
                          channel_axis=3, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, K1=0.01, K2=0.03))
        
        
    # select a sample of input images
    [X_realB, X_realA], _ = generate_real_samples(n_patches_val, val_dir, img_shape, n_samples, 1)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    # scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
    
   
    # plot real source images
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(np.dstack((np.max(X_realA[i],axis=0), np.zeros((128, 128)))))
    # plot generated target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(np.dstack((np.max(X_fakeB[i],axis=0), np.zeros((128, 128)))))
    # plot real target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
        pyplot.axis('off')
        pyplot.imshow(np.dstack((np.max(X_realB[i],axis=0), np.zeros((128, 128)))))
    # save plot to file
    filename1 = 'plot_%06d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model_%06d.h5' % (step+1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))

    
    return np.mean(np.asarray(ssim_aux))

# train Vox2Vox model
def train(d_model, g_model, gan_model, img_shape, train_dir, val_dir, n_epochs=200, n_batch=2):    
    # validation patches and training patches
    n_patches_val = len(os.listdir(os.path.join(val_dir, 'inputs/')))-1
    n_patches_train = len(os.listdir(os.path.join(train_dir, 'inputs/')))-1
    print('Number of Training Samples: %i' % n_patches_train)
    print('Number of Validation Samples: %i' % n_patches_val)	

    # determine the output square shape of the discriminator
    losses_list = []
    ssim_values = []
    epochs_ssim = []
    n_patch = d_model.output_shape[2]

    # number of batches per epoch
    bat_per_epo = n_patches_train // n_batch  # Ensure full batches
    n_steps = bat_per_epo * n_epochs
    i = 0  # step counter

    for k in range(n_epochs):
        # shuffle indices for training samples
        array_samples = np.arange(n_patches_train)
        np.random.shuffle(array_samples)

        # iterate over batches
        for batch_idx in range(bat_per_epo):
            # Get batch indices
            batch_samples = array_samples[batch_idx * n_batch:(batch_idx + 1) * n_batch]
            #print('batch {}'.format(batch_samples))

            # Generate real samples for the batch
            [X_realB, X_realA], y_real = generate_real_samples2(batch_samples, n_batch, n_patch, train_dir, img_shape)

            # Generate fake samples for the batch
            X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)

            # Update discriminator for real samples
            d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)

            # Update discriminator for fake samples
            d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)

            # Update the generator
            g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])

            # Print loss summary
            print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
            losses_list.append([i+1, d_loss1, d_loss2, g_loss])

            # Evaluate model every 5 epochs
            if (i+1) % (bat_per_epo * 5) == 0:
                epochs_ssim.append(i)
                ssim_ = summarize_performance(i, g_model, n_patches_val, val_dir, img_shape)
                ssim_values.append(ssim_)
            
            i += 1  # Increment step counter

    # Save losses and validation results
    np.save('listlosses.npy', losses_list)
    df = pd.DataFrame({'Step': epochs_ssim, 'SSIM': ssim_values})
    df.to_csv(r'eval_validation.csv')