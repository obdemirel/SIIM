import tensorflow as tf
import scipy.io as sio
import numpy as np
import time
from datetime import datetime
import os
import h5py as h5
import hdf5storage
import matplotlib.pyplot as plt
import pdb
from tensorflow.python.client import device_lib

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
nb_blocks = 10 #Unrolled block number
epochs = 100 #Number of epochs
batchSize = 1 #Batch size of 1 is used
num_res_blocks = 15  # ResNet block number
acc_rate = 4 #In-plane acceleration rate, dummy for this code
# Parameters
learning_rate = 3e-4 #Learning Rate
LR = 'LR_3e4_'

num_gpus = [0] #assign the GPU, for multi GPU plase use [0,1] etc.

#need to select wich training is in process
#ESPIRiT: conentional encoding operator is used
#Unnorm: Signal-intensity informed encoding operator (SSIM) is used
espirit = False
unnorm = True

rho = 0.4  # Lambda-Omega ratio
num_reps = 5  # Multi-mask Number
data_opt = 'DL_perf'  # to selct the data_option, can be considered as dummy for now

#To select themasking option. One can choose to use Gaussian but uniform works better
if num_reps == 1:
    # mask_type ='Gaussian' usually gaussian is better in this setting
    mask_type = 'Uniform'
else:
    mask_type = 'Uniform'
directory_suffix = mask_type + '_Random' + str(np.int(rho * 100)) + 'Perc_' + str(
    num_reps) + 'Reps'  # 'Gaussian_x2y2_' + str(np.int(rho*100)) + 'Perc'#'Uniform_' + str(num_reps) + 'Reps_UniformRandom'
print('data opt: ', data_opt, ', acc rate : ', acc_rate, ', mask type :', mask_type, ', num reps: ', num_reps)

#In case you want to start from a pre-trained model, please use True
transfer_learning_option = False

if data_opt == 'DL_perf':
    
    #Slice Number, RO, PE, Channel
    slice_size, nrow_GLOB, ncol_GLOB, ncoil_GLOB = 3, 164, 172, 34 

    if transfer_learning_option:
        data_tag = 'perf_TL_'  ## just naming as TL to make a difference
        TL_path = './savedModels/my_model'  # set the directory of the pre-trained model
    else:
        data_tag = 'perf_'

    ### DATA LOADING PART ######################################################
    file_dir = "/home/daedalus1-raid1/omer-data/perfusion_github/database/subject_"  # choose the database path

    maps_list = list()
    rssq_maps_list = list()
    kspace_list = list()
    padded_mask_list = list()
    unpadded_mask_list = list()
    warm_start_list = list()
    subjects = [1] # only have 1 subject for this basic example

    for subject_num in subjects:
        slices = [1] #only have 1 slab group for this basic example

        for slice_num in slices:
            print ('Subject Num: ', subject_num)
            kspace = hdf5storage.loadmat(file_dir + str(subject_num)  + ".mat")['kspace_all']
            maps = hdf5storage.loadmat(file_dir + str(subject_num)  + ".mat")['sense_maps_all']

            padded_mask = hdf5storage.loadmat(file_dir + str(subject_num)  + ".mat")['mask_all']
            unpadded_mask = hdf5storage.loadmat(file_dir + str(subject_num)  + ".mat")['mask_all']
            intensity = hdf5storage.loadmat(file_dir + str(subject_num)  + ".mat")['sense_maps_intensity_all']

            kspace = np.transpose(np.copy(kspace[:, :, :, :]), axes=(3, 0, 1, 2))  

            maps = np.tile(maps[:, :, :, :, np.newaxis], (1, 1, 1, 1, intensity.shape[3])) # need to repeat ESPIRiT maps for each time-frame to be multiplied by intensities
            maps = np.transpose(np.copy(maps[:, :, :, :, :]), axes=(4, 3, 0, 1, 2))

            if (unnorm):
                intensity = np.transpose(np.copy(intensity[:, :, :, :]), axes=(3, 2, 0, 1))
                intensity = np.tile(intensity[:, :, :, :, np.newaxis], (1, 1, 1, 1, 34))
                maps = maps * intensity
                
                #need to normalize SSIM maps since they are not in [0,1]
                for ii in range(maps.shape[0]):
                    for kk in range(maps.shape[1]):
                        tmp = maps[ii, kk, ...]
                        tmp = tmp / np.max(np.abs(tmp[:]))
                        maps[ii,kk,...] = tmp * 1

            padded_mask = np.transpose(np.copy(padded_mask[:, :, :, :]), axes=(3, 0, 1, 2))
            unpadded_mask = np.transpose(np.copy(unpadded_mask[:, :, :, :]), axes=(3, 0, 1, 2))

            print('kspace: ', kspace.shape, ', maps: ', maps.shape)
            print('padded mask: ', padded_mask.shape, ', unpadded mask: ', unpadded_mask.shape)


            maps_list.append(maps)
            kspace_list.append(kspace)
            padded_mask_list.append(padded_mask[..., 0])
            unpadded_mask_list.append(unpadded_mask[..., 0])
            print('kspace: ', np.concatenate(np.asarray(kspace_list), axis=0).shape, ', maps: ',
                  np.concatenate(np.asarray(maps_list), axis=0).shape, ',  padded mask : ',
                  np.concatenate(np.asarray(padded_mask_list), axis=0).shape)

            kspace_train = np.concatenate(np.asarray(kspace_list), axis=0)
            trnCsm = np.concatenate(np.asarray(maps_list), axis=0)
            padded_mask = np.concatenate(np.asarray(padded_mask_list), axis=0)
            unpadded_mask = np.concatenate(np.asarray(unpadded_mask_list), axis=0)
            ############################################################################
            
            
#Some useful TF function
c2r_tf = lambda x: tf.stack([tf.real(x), tf.imag(x)], axis=-1)
# r2c_tf takes the last dimension of real input and converts to complex
r2c_tf = lambda x: tf.complex(x[..., 0], x[..., 1])


#Mask geeration for SSDU
def create_mask(input_data, padded_mask, unpadded_mask, mask_option='Uniform', num_reps=10, rho=0.1, num_iter=0):
    [nx, ny] = padded_mask.shape

    if mask_option == 'Uniform':
        mask_val = np.zeros_like(unpadded_mask)
        temp_mask = np.copy(unpadded_mask)
        mx = int(find_center_ind(input_data, axes=(1, 2)))
        my = int(find_center_ind(input_data, axes=(0, 2)))
        if num_iter == 0:
            print('center of kspace, mx: ', mx, ', my: ', my)
        temp_mask[mx - 2: mx + 2, my - 2: my + 2] = 0
        pr = np.ndarray.flatten(temp_mask)
        ind = np.random.choice(np.arange(nx * ny),
                               size=np.int(np.count_nonzero(pr) * rho), replace=False, p=pr / np.sum(pr))

        [ind_x, ind_y] = index_flatten2nd(ind, (nx, ny))
        mask_val[ind_x, ind_y] = 1
        mask_trn = padded_mask - mask_val
        return padded_mask, mask_trn, mask_val

    if mask_option == 'Gaussian':
        count = 0
        test_pts = np.int(np.ceil(np.sum(unpadded_mask[:]) * rho))
        mask_val = np.zeros_like(unpadded_mask)
        temp_mask = np.copy(unpadded_mask)
        mx = int(find_center_ind(input_data, axes=(1, 2)))
        my = int(find_center_ind(input_data, axes=(0, 2)))
        if num_iter == 0:
            print('center of kspace, mx: ', mx, ', my: ', my)
        temp_mask[mx - 2: mx + 2, my - 2: my + 2] = 0
        while count <= test_pts:
            indx = np.int(np.round(np.random.normal(loc=mx, scale=(nx - 1) / 2)))
            indy = np.int(np.round(np.random.normal(loc=my, scale=(ny - 1) / 2)))
            if (0 <= indx < nx and 0 <= indy < ny and temp_mask[indx, indy] == 1 and mask_val[indx, indy] != 1):
                mask_val[indx, indy] = 1
                count = count + 1
        mask_trn = padded_mask - mask_val
        return padded_mask, mask_trn, mask_val

# Some useful functions
def index_flatten2nd(ind, shape):
    array = np.zeros(np.prod(shape))
    array[ind] = 1
    ind_nd = np.nonzero(np.reshape(array, shape))
    return [list(ind_nd_ii) for ind_nd_ii in ind_nd]


def c2r(inp):
    """  input img: row x col in complex64
    output image: row  x col x2 in float32
    """
    if inp.dtype == 'complex64':
        dtype = np.float32
    else:
        dtype = np.float64
    out = np.zeros(inp.shape + (2,), dtype=dtype)
    out[..., 0] = inp.real
    out[..., 1] = inp.imag
    return out


def r2c(inp):
    """  input img: row x col x 2 in float32
    output image: row  x col in complex64
    """
    if inp.dtype == 'float32':
        dtype = np.complex64
    else:
        dtype = np.complex128
    out = np.zeros(inp.shape[0:2], dtype=dtype)
    out = inp[..., 0] + 1j * inp[..., 1]
    return out


def fft(ispace, axes=(0, 1), norm=None, unitary_opt=True):
    kspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(ispace, axes=axes), axes=axes, norm=norm), axes=axes)

    if unitary_opt:
        fact = 1
        for axis in axes:
            fact = fact * kspace.shape[axis]
        kspace = kspace / np.sqrt(fact)

    return kspace


def ifft(kspace, axes=(0, 1), norm=None, unitary_opt=True):
    ispace = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(kspace, axes=axes), axes=axes, norm=norm), axes=axes)

    if unitary_opt:
        fact = 1
        for axis in axes:
            fact = fact * ispace.shape[axis]
        ispace = ispace * np.sqrt(fact)

    return ispace


def tf_flip2D(input_data, axes=1):
    nx = int(nrow_GLOB / 2)
    ny = int(ncol_GLOB / 2)
    nc = ncoil_GLOB
    if axes == 1:
        first_half = tf.identity(input_data[:, :nx, :])
        second_half = tf.identity(input_data[:, nx:, :])
    elif axes == 2:
        first_half = tf.identity(input_data[:, :, :ny])
        second_half = tf.identity(input_data[:, :, ny:])
    else:
        raise ValueError('Invalid nums of dims')
    return tf.concat([second_half, first_half], axis=axes)


def tf_fftshift(input_x):
    nx = int(nrow_GLOB / 2)
    ny = int(ncol_GLOB / 2)
    p1 = input_x
    first_half = tf.identity(p1[..., 0:nx, :])
    second_half = tf.identity(p1[..., nx:, :])
    p2 = tf.concat([second_half, first_half], 1)
    first_half = tf.identity(p2[..., :, 0:ny])
    second_half = tf.identity(p2[..., :, ny:])
    p3 = tf.concat([second_half, first_half], 2)
    return p3


def tf_fftshift2(input_x):
    nx = int(162 / 2)
    ny = int(170 / 2)
    p1 = input_x
    first_half = tf.identity(p1[..., 0:nx, :])
    second_half = tf.identity(p1[..., nx:, :])
    p2 = tf.concat([second_half, first_half], 1)
    first_half = tf.identity(p2[..., :, 0:ny])
    second_half = tf.identity(p2[..., :, ny:])
    p3 = tf.concat([second_half, first_half], 2)
    return p3


def norm(tensor, axes=(0, 1, 2), keepdims=True):
    for axis in axes:
        tensor = np.linalg.norm(tensor, axis=axis, keepdims=True)
    if not keepdims: return tensor.squeeze()
    return tensor


def find_center_ind(kspace, axes=(1, 2, 3)):
    pow = norm(kspace, axes=axes).squeeze()
    return np.argsort(pow)[-1:]


def conv_layer(x, szW, is_training, is_relu, is_scaling):
    """
    This function create a layer of CNN consisting of convolution,ReLU&scaling
    """
    W = tf.get_variable('W', shape=szW,
                        initializer=tf.random_normal_initializer(0, 0.05))  # tf.contrib.layers.xavier_initializer())
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    # xbn=tf.layers.batch_normalization(x,training=is_training,fused=True,name='BN')

    if (is_relu):
        x = tf.nn.relu(x)
    if (is_scaling):
        scalar = tf.constant(0.1, dtype=tf.float32)
        x = tf.multiply(scalar, x)
    return x


def ResNet(inp, is_training, num_res_block):
    """
    This is ResNet Block
    It creates an n  residual blocks.
    Convolution filters are of size 3x3 and 64 such filters are there.
    """
    nw = {}
    szW = {}
    szW[1] = (3, 3, 2, 64)  # Convolution at the input
    szW[2] = (3, 3, 64, 64)  # convolution during the residual blocks
    szW[3] = (3, 3, 64, 2)  # convolutions at the output
    with tf.variable_scope('FirstLayer'):
        nw['c0'] = conv_layer(inp, szW[1], is_training, is_relu=False, is_scaling=False)

    for i in np.arange(1, num_res_block + 1):
        with tf.variable_scope('ResBlock' + str(i)):
            conv_layer1 = conv_layer(nw['c' + str(i - 1)], szW[2], is_training, is_relu=True, is_scaling=False)
            conv_layer2 = conv_layer(conv_layer1, szW[2], is_training, is_relu=False, is_scaling=True)
            nw['c' + str(i)] = conv_layer2 + nw['c' + str(i - 1)]

    with tf.variable_scope('LastLayer'):
        rb_output = conv_layer(nw['c' + str(i)], szW[2], is_training, is_relu=False, is_scaling=False)

    with tf.variable_scope('LastLayer2'):
        temp_output = rb_output + nw['c0']
        nw_output = conv_layer(temp_output, szW[3], is_training, is_relu=False, is_scaling=False)

    with tf.name_scope('Residual'):
        dw = tf.identity(nw_output)  # tf.identity(inp)
        # dw=shortcut+nw_output
    return dw


# Build the function to average the gradients
def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


# By default, all variables will be placed on '/gpu:0'
# So we need a custom device function, to assign all variables to '/cpu:0'
# Note: If GPUs are peered, '/gpu:0' can be a faster option
PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']


def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device

    return _assign


def getLambda():
    """
    create a shared variable called lambda.
    """
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        lam = tf.get_variable(name='lam1', dtype=tf.float32, initializer=.005)  # 0.05 is original
    return lam


# This is the encoding part of the DC
# Since this is a SMS factor 3 acqusition, we have 3 k-spaces.
# Ill-written code but works. One can convert this to a for loop by using a slice acceleration variable
# for now it is left as it is
class Aclass:
    """
    This class is created to do the data-consistency (DC) step as described in paper.
    """

    def __init__(self, csm, mask, lam):
        with tf.name_scope('Ainit'):
            s = tf.shape(mask)
            self.nrow, self.ncol = s[0], s[1]
            print('nrow:', s[0], 'ncol:', s[1])
            self.pixels = self.nrow * self.ncol
            self.mask = mask
            self.csm = csm
            self.SF = tf.complex(tf.sqrt(tf.to_float(self.pixels)), 0.)
            self.lam = lam
            # self.cgIter=cgIter
            # self.tol=tol

    def myAtA(self, img):
        with tf.name_scope('AtA'):
            coilImages1 = self.csm[0, :, :, :] * img[:nrow_GLOB, :]
            coilImages2 = self.csm[1, :, :, :] * img[nrow_GLOB:nrow_GLOB * 2, :]
            coilImages3 = self.csm[2, :, :, :] * img[nrow_GLOB * 2:nrow_GLOB * 3, :]


            kspace1 = tf_fftshift(tf.fft2d(tf_fftshift(coilImages1))) / self.SF
            temp1 = kspace1 * self.mask
            # ...........................................................................................................
            kspace2 = tf_fftshift(tf.fft2d(tf_fftshift(coilImages2))) / self.SF
            temp2 = kspace2 * self.mask
            # ...........................................................................................................
            kspace3 = tf_fftshift(tf.fft2d(tf_fftshift(coilImages3))) / self.SF
            temp3 = kspace3 * self.mask
            # ...........................................................................................................
            temp = temp1 + temp2 + temp3
            # ...........................................................................................................
            coilImgs1 = tf_fftshift(tf.ifft2d(tf_fftshift(temp))) * self.SF
            coilComb1 = tf.reduce_sum(coilImgs1 * tf.conj(self.csm[0, :, :, :]), axis=0)
            coilComb1 = coilComb1 + self.lam * img[:nrow_GLOB, :]
            # ...........................................................................................................
            coilImgs2 = tf_fftshift(tf.ifft2d(tf_fftshift(temp))) * self.SF
            coilComb2 = tf.reduce_sum(coilImgs2 * tf.conj(self.csm[1, :, :, :]), axis=0)
            coilComb2 = coilComb2 + self.lam * img[nrow_GLOB:nrow_GLOB * 2, :]
            # ...........................................................................................................
            coilImgs3 = tf_fftshift(tf.ifft2d(tf_fftshift(temp))) * self.SF
            coilComb3 = tf.reduce_sum(coilImgs3 * tf.conj(self.csm[2, :, :, :]), axis=0)
            coilComb3 = coilComb3 + self.lam * img[nrow_GLOB * 2:nrow_GLOB * 3, :]
            # ...........................................................................................................

            coilComb = tf.concat([coilComb1, coilComb2, coilComb3], axis=0)

        return coilComb

#Conventional encoding operator uses this function with 20 conjugate gradient steps
def myCG(A, rhs, wim, lam2):
    # This is my implementation of CG algorithm in tensorflow that works on
    # complex data and runs on GPU. It takes the class object as input.
    rhs = r2c_tf(rhs)
    wim = r2c_tf(wim)
    cond = lambda i, *_: tf.less(i, 20)

    def body(i, rTr, x, r, p):
        with tf.name_scope('cgBody'):
            Ap = A.myAtA(p)
            alpha = rTr / tf.to_float(tf.reduce_sum(tf.conj(p) * Ap))
            alpha = tf.complex(alpha, 0.)
            x = x + alpha * p
            r = r - alpha * Ap
            rTrNew = tf.to_float(tf.reduce_sum(tf.conj(r) * r))
            beta = rTrNew / rTr
            beta = tf.complex(beta, 0.)
            p = r + beta * p
        return i + 1, rTrNew, x, r, p

    # x = tf.zeros_like(rhs)
    x = wim
    # i, r, p = 0, rhs, rhs
    i, r, p = 0, rhs - A.myAtA(x) * 1, rhs - A.myAtA(x) * 1
    rTr = tf.to_float(tf.reduce_sum(tf.conj(r) * r), )
    loopVar = i, rTr, x, r, p
    out = tf.while_loop(cond, body, loopVar, name='CGwhile', parallel_iterations=1)[2]
    cg_out = out
    return c2r_tf(cg_out)

#SSIM encoding operator uses this function with 20 conjugate gradient steps
#Mm is the preconditoner part of the conjugate gradient operation
def myPCG(A, rhs, Mm, wim):
    """
    This is my implementation of CG algorithm in tensorflow that works on
    complex data and runs on GPU. It takes the class object as input.
    """

    rhs = r2c_tf(rhs)
    wim = r2c_tf(wim)
    Mm = r2c_tf(Mm)
    Mm = 1 / Mm
    cond = lambda i, *_: tf.less(i, 20)

    def body(i, rTz, x, r, p, z):
        with tf.name_scope('cgBody'):
            Ap = A.myAtA(p)
            alpha = rTz / tf.to_float(tf.reduce_sum(tf.conj(p) * Ap))
            alpha = tf.complex(alpha, 0.)
            x = x + alpha * p
            r = r - alpha * Ap
            z = Mm * r
            rTzNew = tf.to_float(tf.reduce_sum(tf.conj(r) * z))
            beta = rTzNew / rTz
            beta = tf.complex(beta, 0.)
            p = z + beta * p
        return i + 1, rTzNew, x, r, p, z

    # x = tf.zeros_like(rhs)
    x = wim
    # i, r, p, z = 0, rhs, rhs, Mm * rhs
    i, r, p, z = 0, rhs - A.myAtA(x), Mm * (rhs - A.myAtA(x)), Mm * (rhs - A.myAtA(x))
    # i, r, p , z = 0, rhs, rhs, tf.reshape(Mm,[nrow_GLOB*ncol_GLOB])*rhs
    # rTr = tf.to_float(tf.reduce_sum(tf.conj(r) * r), )
    p = z
    rTz = tf.to_float(tf.reduce_sum(tf.conj(r) * z), )
    loopVar = i, rTz, x, r, p, z
    out = tf.while_loop(cond, body, loopVar, name='CGwhile', parallel_iterations=1)[2]
    return c2r_tf(out)

#encoder transpose operation for data consistency step
class Uclass:
    """
    This class is created to do the data-consistency (DC) step as described in paper.
    """

    def __init__(self, csm, mask):
        with tf.name_scope('Uinit'):
            s = tf.shape(mask)
            self.nrow, self.ncol = s[0], s[1]
            print('shape of mask', tf.shape(mask))

            self.pixels = self.nrow * self.ncol
            self.mask = mask
            self.csm = csm
            self.SF = tf.complex(tf.sqrt(tf.to_float(self.pixels)), 0.)

    def myUlAtA(self, img):
        with tf.name_scope('ULAtA'):
            coilImages1 = self.csm[0, :, :, :] * img[:nrow_GLOB, :]
            coilImages2 = self.csm[1, :, :, :] * img[nrow_GLOB:nrow_GLOB * 2, :]
            coilImages3 = self.csm[2, :, :, :] * img[nrow_GLOB * 2:nrow_GLOB * 3, :]

            kspace1 = tf_fftshift(tf.fft2d(tf_fftshift(coilImages1))) / self.SF
            temp1 = kspace1 * self.mask
            # ...........................................................................................................
            kspace2 = tf_fftshift(tf.fft2d(tf_fftshift(coilImages2))) / self.SF
            temp2 = kspace2 * self.mask
            # ...........................................................................................................
            kspace3 = tf_fftshift(tf.fft2d(tf_fftshift(coilImages3))) / self.SF
            temp3 = kspace3 * self.mask
            # ...........................................................................................................

            temp = temp1 + temp2 + temp3
        return temp


def dc_unsupervised(rhs, csm, mask):
    """
    This function is called to create testing model. It apply CG on each image
    in the batch.
    """
    # lam2=tf.complex(lam1,0.)
    rhs = r2c_tf(rhs)

    def fn(tmp):
        c, m, r = tmp
        Aobj = Uclass(c, m)
        y = Aobj.myUlAtA(r)
        return y

    inp = (csm, mask, rhs)
    rec = tf.map_fn(fn, inp, dtype=tf.complex64, name='valmapFn')
    return c2r_tf(rec)


def dc(rhs, csm, mask, lam1, warm_image):
    """
    This function is called to create testing model. It apply CG on each image
    in the batch.
    """
    lam2 = tf.complex(lam1, 0.)

    def fn(tmp):
        c, m, r, wim = tmp
        Aobj = Aclass(c, m, lam2)
        y = myCG(Aobj, r, wim, lam2)
        return y

    inp = (csm, mask, rhs, warm_image)
    rec = tf.map_fn(fn, inp, dtype=tf.float32, name='mapFn')
    return rec



def dc_pcg(rhs, csm, mask, lam1, Mmat, fixV, warm_image):
    """
    This function is called to create testing model. It apply CG on each image
    in the batch.
    """

    lam2 = tf.complex(lam1, 0.)
    fixV2 = tf.complex(fixV, 0.)  # * 1000 / 2

    def fn(tmp):
        c, m, r, Mm, wim = tmp
        Aobj = Aclass(c, m, lam2)

        ##calcualte the new Mm with thk
        Mm = c2r_tf(r2c_tf(Mm) + lam2 * fixV2 * tf.ones([nrow_GLOB * slice_size, ncol_GLOB], dtype=tf.complex64))
        ##################

        y = myPCG(Aobj, r, Mm, wim)
        return y

    inp = (csm, mask, rhs, Mmat, warm_image)
    rec = tf.map_fn(fn, inp, dtype=tf.float32, name='mapFn')
    return rec


# Each SMS slice has a FOV/3 CAIPI shift
# This function shift appropriate amount of FOV shifts for each slice
# This is neccessary for SMS encoding, otherwise CNN will see shifted images where
# boundary artifacts would be the issue
# Shift_amo determines the shifting amount of the FOV in terms of pixels
def shifterb(input_data, shift_amo):

    f3 = tf.identity(input_data[..., nrow_GLOB * 2:3 * nrow_GLOB, :, :])

    first_half2 = tf.identity(
        input_data[..., 0:1 * nrow_GLOB, :1 * shift_amo - 2, :])  # -2 for 172 PE, 0 for PF included
    second_half2 = tf.identity(input_data[..., 0:1 * nrow_GLOB, 1 * shift_amo - 2:, :])
    first_half3 = tf.identity(input_data[..., nrow_GLOB * 1:2 * nrow_GLOB, :shift_amo * 2 - 2, :])
    second_half3 = tf.identity(input_data[..., nrow_GLOB * 1:2 * nrow_GLOB, shift_amo * 2 - 2:, :])


    f1 = tf.concat([second_half2, first_half2], axis=2)
    f2 = tf.concat([second_half3, first_half3], axis=2)

    return tf.concat([f1, f2, f3], axis=1)

# After the CNN, we need to reshift the CAIPIRINHA FOVs for SMS encoding
def shifterf(input_data, shift_amo):
    f3 = tf.identity(input_data[..., nrow_GLOB * 2:3 * nrow_GLOB, :, :])

    first_half2 = tf.identity(input_data[..., 0:1 * nrow_GLOB, :2 * shift_amo, :])
    second_half2 = tf.identity(input_data[..., 0:1 * nrow_GLOB, 2 * shift_amo:, :])
    first_half3 = tf.identity(input_data[..., nrow_GLOB * 1:2 * nrow_GLOB, : 1 * shift_amo, :])
    second_half3 = tf.identity(input_data[..., nrow_GLOB * 1:2 * nrow_GLOB, 1 * shift_amo:, :])

    f1 = tf.concat([second_half2, first_half2], axis=2)
    f2 = tf.concat([second_half3, first_half3], axis=2)

    return tf.concat([f1, f2, f3], axis=1)


#Unrolling the network starts here
class UnrolledNet():
    def __init__(self, input_x, sens_maps, mask, mask_val, nb_blocks, num_res_blocks, Ms, FV, wrm, training):
        self.input_x = input_x
        self.sens_maps = sens_maps
        self.mask = mask
        self.mask_val = mask_val
        self.nb_blocks = nb_blocks
        self.num_res_blocks = num_res_blocks
        self.Ms = Ms
        self.FV = FV
        self.wrm = wrm
        self.training = training
        self.model = self.Unrolled()

    def Unrolled(self):

        x, denoiser_output, dc_output = self.input_x, self.input_x, self.wrm
        #################################################################################
        all_intermediate_results = [[0 for _ in range(2)] for _ in range(self.nb_blocks)]
        lam_init = tf.constant(0.05, dtype=tf.float32)
        x0 = self.input_x
        x = x0
        dc_output = x0
        #################################################################################

        with tf.name_scope('myModel'):
            with tf.variable_scope('Wts', reuse=tf.AUTO_REUSE):
                for i in range(self.nb_blocks):
                    x = shifterf(x, 58)  # used to be 58 for 172 PE
                    x = ResNet(x, self.training, self.num_res_blocks)
                    denoiser_output = x
                    x = shifterb(x, 58)

                    lam1 = getLambda()

                    rhs = self.input_x + ((lam1 * 1) * x)

                    if (unnorm):
                        x = dc_pcg(rhs, self.sens_maps, self.mask, lam1, self.Ms, self.FV, dc_output)
                    if (espirit):
                        x = dc(rhs, self.sens_maps, self.mask, lam1, dc_output)

                    dc_output = x

                    # ...................................................................................................
                    all_intermediate_results[i][0] = r2c_tf(tf.squeeze(denoiser_output))
                    all_intermediate_results[i][1] = r2c_tf(tf.squeeze(dc_output))

            ul_output = dc_unsupervised(x, self.sens_maps, self.mask_val)
        return x, ul_output, x0, all_intermediate_results, lam1

# SENSE-1 recon using sensitivity maps
def sense1(input_kspace, sens_maps):
    [m, n, nc] = np.shape(sens_maps)
    image_space = ifft(input_kspace, axes=(0, 1), norm=None, unitary_opt=True)
    Eh_op = np.conj(sens_maps) * image_space
    Eh_op = np.sum(Eh_op, axis=2)

    return Eh_op


# This is used for the preconditioner
def fixval(cc):
    im_space = ifft(cc, axes=(0, 1), norm=None, unitary_opt=True)
    rssq_im = np.sqrt(np.sum(np.square(np.abs(im_space)), axis=2))

    thkfix = np.max(rssq_im.flatten())

    return thkfix

# Preconditioner matrix calculation using maps
def Mcal(sens_maps, maskp, thk):
    [m, n, nc] = np.shape(sens_maps)
    M = np.empty((m, n), dtype=np.complex64)
    p1 = np.sum(np.abs(np.square(sens_maps)), axis=2)
    p2 = np.sum(np.sum(np.sum(maskp, axis=2), axis=1), axis=0) / (m * n * nc)

    M = (p1 * p2)
    return M

#If you have pre-trained model to start
if transfer_learning_option:
    cwd = os.getcwd()
    print('Getting weights from trained model:')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    loadChkPoint_tl = TL_path + '/model-99' #place the latest epoch are the best epoch
    with tf.Session(config=config) as sess:
        new_saver = tf.train.import_meta_graph(TL_path + '/modelTst.meta')
        new_saver.restore(sess, loadChkPoint_tl)
        trainable_collection_trained = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
        nontrainable_variables_trained = [sess.run(v) for v in trainable_collection_trained]
        print(len(nontrainable_variables_trained))
        print('\n\n\n')
    print('Trained model is loaded')


#Initilazitation of the network
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
# --------------------------------------------------------------------------
# %%Generate a meaningful filename to save the trainined models for testing
print('*************************************************')
start_time = time.time()
saveDir = 'savedModels/'
cwd = os.getcwd()  # returns current working directory of a process
directory = saveDir + \
            data_tag + str(acc_rate) + 'R_' + str(nb_blocks) + 'K_' + str(num_res_blocks) + 'RB_' + str(
    epochs) + 'E_' + LR \
            + directory_suffix  # 'Uniform_' + str(num_reps) + 'Reps_UniformRandom'

if not os.path.exists(directory):
    os.makedirs(directory)
sessFileName = directory + '/model'

# %% save test model
tf.reset_default_graph()
csmT = tf.placeholder(tf.complex64, shape=(None, slice_size, ncoil_GLOB, nrow_GLOB, ncol_GLOB), name='csm')
maskT = tf.placeholder(tf.complex64, shape=(None, nrow_GLOB, ncol_GLOB), name='mask')
maskV = tf.placeholder(tf.complex64, shape=(None, nrow_GLOB, ncol_GLOB), name='testmaskV')
atbT = tf.placeholder(tf.float32, shape=(None, nrow_GLOB * slice_size, ncol_GLOB, 2), name='atb')
MsT = tf.placeholder(tf.float32, shape=(None, nrow_GLOB * slice_size, ncol_GLOB, 2), name='mst')
FVT = tf.placeholder(tf.float32, shape=(None,), name='fvt')
WarmT = tf.placeholder(tf.float32, shape=(None, nrow_GLOB * slice_size, ncol_GLOB, 2), name='warmt')
nw_out, ul_output, x0, all_intermediate_outputs, lam = UnrolledNet(atbT, csmT, maskT, maskV, nb_blocks, num_res_blocks,
                                                                   MsT, FVT, WarmT,
                                                                   False).model  # False for is_training(BN)

out = tf.identity(nw_out, name='out')
ul_output = tf.identity(ul_output, name='predTst')
all_intermediate_outputs = tf.identity(all_intermediate_outputs, name='all_intermediate_outputs')
x0 = tf.identity(x0, name='x0')
lam = tf.identity(lam, name='lam')
sessFileNameTst = directory + '/modelTst'

saver = tf.train.Saver()
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    savedFile = saver.save(sess, sessFileNameTst, latest_filename='checkpointTst')
print('testing model saved:' + savedFile)

# .......................................................................................................................
print('Loading training data...')
print('size of the training data', np.shape(kspace_train))

print('Normalize the kspace to 0-1 region')
for ii in range(np.shape(kspace_train)[0]):
    #k-space normalization is done
    temp = np.copy(kspace_train[ii, ...])
    kspace_train[ii, ...] = temp / np.max(np.abs(temp[:]))

    for kk in range(slice_size):
        temp = np.copy(trnCsm[ii, kk, ...])
        trnCsm[ii, kk, ...] = temp
    if np.max(np.abs(temp[:])) == 0:
        print('Max is zero at Iter ', ii)
        raise ValueError('Max is zero')


print('size of the training data', kspace_train.shape, ', coil maps: ', trnCsm.shape)

nSlice, nrow, ncol, ncoil = kspace_train.shape

origMask, trnMask, valMask = np.empty((nSlice, num_reps, nrow, ncol), dtype=np.complex64), \
                             np.empty((nSlice, num_reps, nrow, ncol), dtype=np.complex64), \
                             np.empty((nSlice, num_reps, nrow, ncol), dtype=np.complex64)

trnAtb = np.empty((nSlice, num_reps, nrow * slice_size, ncol), dtype=np.complex64)
ref_kspace = np.empty((nSlice, num_reps, nrow, ncol, ncoil), dtype=np.complex64)
trnCsmAll = np.empty((nSlice, num_reps, slice_size, nrow, ncol, ncoil), dtype=np.complex64)
trnM = np.empty((nSlice, num_reps, nrow * slice_size, ncol), dtype=np.float32)
trnFixVal = np.empty((nSlice, num_reps), dtype=np.float32)
trnWarmAll = np.empty((nSlice, num_reps, nrow * slice_size, ncol), dtype=np.complex64)

# .......................................
print('Multi Mask Version -- getting the refs and aliased sense1 images')
for ii in range(nSlice):
    if np.mod(ii, 15) == 0:
        print('Iteration: ', ii)
    for jj in range(num_reps):
        origMask[ii, jj, ...], trnMask[ii, jj, ...], valMask[ii, jj, ...] = create_mask(kspace_train[ii],
                                                                                        padded_mask[ii, ...],
                                                                                        unpadded_mask[ii, ...],
                                                                                        mask_option=mask_type, rho=rho,
                                                                                        num_iter=ii)
        proc_mask = np.copy(trnMask[ii, jj, ...])
        proc_maskV = np.copy(valMask[ii, jj, ...])
        proc_mask = np.tile(proc_mask[:, :, np.newaxis], (1, 1, ncoil))
        proc_maskV = np.tile(proc_maskV[:, :, np.newaxis], (1, 1, ncoil))
        sub_kspace = kspace_train[ii] * proc_mask
        ref_kspace[ii, jj, ...] = kspace_train[ii] * proc_maskV
        for kk in range(slice_size):
            idx_start, idx_end = kk * nrow_GLOB, (kk + 1) * nrow_GLOB
            trnAtb[ii, jj, idx_start:idx_end, ...] = sense1(sub_kspace, trnCsm[ii, kk, ...])

            trnM[ii, jj, idx_start:idx_end, ...] = Mcal(trnCsm[ii, kk, ...], proc_mask, 1e-7)


        trnFixVal[ii, jj] = fixval(kspace_train[ii, ...])

        trnCsmAll[ii, jj, ...] = np.copy(trnCsm[ii, ...])
        trnWarmAll[ii, jj, ...] = np.copy(trnAtb[ii, jj, ...])

trnWarm = np.concatenate(list(trnWarmAll), axis=0)
trnCsm = np.concatenate(list(trnCsmAll), axis=0)
ref_kspace = np.concatenate(list(ref_kspace), axis=0)
trnMask = np.concatenate(list(trnMask), axis=0)
valMask = np.concatenate(list(valMask), axis=0)
trnAtb = np.concatenate(list(trnAtb), axis=0)
trnM = np.concatenate(list(trnM), axis=0)
trnFixVal = np.concatenate(list(trnFixVal), axis=0)

sio.savemat(('unnormalized_Sense1_images.mat'), {'input': trnAtb})
sio.savemat(('warm_ins.mat'), {'warm': trnWarm})
sio.savemat(('valmasks.mat'), {'trnMask': trnMask, 'valMask': valMask})
print('size of ref kspace: ', np.shape(ref_kspace), ', coil maps: ', trnCsm.shape, ' size of input: ', np.shape(trnAtb), \
      'size of trn mask: ', trnMask.shape, ', size of val mask: ', valMask.shape)


trnAtb = c2r(trnAtb)
trnM = c2r(trnM)
trnWarm = c2r(trnWarm)
trnCsm = np.transpose(trnCsm, (0, 1, 4, 2, 3))
ref_kspace = np.transpose(ref_kspace, (0, 3, 1, 2))
ref_kspace = c2r(ref_kspace)
print(
    'size of reference kspace after c2r: ', np.shape(ref_kspace), ', size of maps: ', trnCsm.shape, ', size of input: ',
    trnAtb.shape)
# %%
# %% creating the dataset
nTrn = trnAtb.shape[0]
total_batch = int(np.floor(np.float32(nTrn) / (batchSize * len(num_gpus))))
nSteps = total_batch * epochs
assert not np.any(np.isnan(trnAtb))
# %%
# Place all ops on CPU by default
tf.reset_default_graph()
with tf.device('/cpu:0'):
    # tf.reset_default_graph()
    tower_grads = []
    kspaceP = tf.placeholder(tf.float32, shape=(None, None, None, None, 2), name='refkspace')
    csmP = tf.placeholder(tf.complex64, shape=(None, slice_size, ncoil_GLOB, nrow_GLOB, ncol_GLOB), name='csm')
    maskP = tf.placeholder(tf.complex64, shape=(None, None, None), name='mask')
    maskVal = tf.placeholder(tf.complex64, shape=(None, None, None), name='maskVal')
    atbP = tf.placeholder(tf.float32, shape=(None, nrow_GLOB * slice_size, ncol_GLOB, 2), name='atb')
    MsP = tf.placeholder(tf.float32, shape=(None, nrow_GLOB * slice_size, ncol_GLOB, 2), name='Ms')
    FVP = tf.placeholder(tf.float32, shape=(None,), name='FV')
    WarmP = tf.placeholder(tf.float32, shape=(None, slice_size * nrow_GLOB, ncol_GLOB, 2), name='Warm')
    # %% creating the dataset
    dataset = tf.data.Dataset.from_tensor_slices((kspaceP, atbP, csmP, maskP, maskVal, MsP, FVP, WarmP))
    # dataset = dataset.cache()
    # dataset=dataset.repeat()
    dataset = dataset.shuffle(buffer_size=10 * len(num_gpus))
    dataset = dataset.batch(batchSize)
    dataset = dataset.prefetch(len(num_gpus))
    iterator = dataset.make_initializable_iterator()
    for i in range(len(num_gpus)):  #
        with tf.device(assign_to_device('/gpu:{}'.format(num_gpus[i]), ps_device='/cpu:0')):
            refT, atbT, csmT, maskT, maskV, MsT, FVT, WarmT = iterator.get_next('getNext')
            # %% make training model
            out, ul_output, _, all_intermediate_results, _ = UnrolledNet(atbT, csmT, maskT, maskV, nb_blocks,
                                                                         num_res_blocks, MsT, FVT, WarmT,
                                                                         True).model  # True indicates is_training
            out = tf.identity(out, name='nw_out')
            predT = tf.identity(ul_output, name='pred')
            scalar = tf.constant(0.5, dtype=tf.float32)
            loss = tf.multiply(scalar, tf.norm(refT - ul_output) / tf.norm(refT)) + tf.multiply(scalar, tf.norm(
                refT - ul_output, ord=1) / tf.norm(refT,
                                                   ord=1))  # tf.reduce_mean(tf.div_no_nan(tf.abs(refT-ul_output),tf.abs(refT)))#
            all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            grads = optimizer.compute_gradients(loss)
            # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            tower_grads.append(grads)
    # training codes
    tower_grads = average_gradients(tower_grads)
    train_op = optimizer.apply_gradients(tower_grads)
    print ('parameters are: Epochs:', epochs, ' BS:', batchSize, 'nblocks:', nb_blocks)
    saver = tf.train.Saver(max_to_keep=100)
    totalLoss, totalTime, ep = [], [], 0
    avg_cost = 0
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        feedDict = {kspaceP: ref_kspace, atbP: trnAtb, maskP: trnMask, maskVal: valMask, csmP: trnCsm, MsP: trnM,
                    FVP: trnFixVal, WarmP: trnWarm}
        # sess.run(iterator.initializer,feed_dict=feedDict)
        # savedFile=saver.save(sess, sessFileName)
        # print("Model meta graph saved in::%s" % savedFile)
        # print('Number of Parameters')
        # print(sess.run(all_trainable_vars))
        if transfer_learning_option:
            print('Assigning weights to new model:')
            # trainable_collection_test = tf.global_variables()
            trainable_collection_test = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
            nontrainable_variables_test = [v for v in trainable_collection_test]
            print(len(nontrainable_variables_test))
            print('\n\n\n')
            for ii in range(len(nontrainable_variables_test)):
                sess.run(nontrainable_variables_test[ii].assign(nontrainable_variables_trained[ii]))
        print('Training...')
        for ii in range(epochs):
            sess.run(iterator.initializer, feed_dict=feedDict)
            if ii == 0:
                savedFile = saver.save(sess, sessFileName)
                print("Model meta graph saved in::%s" % savedFile)
                print('Number of Parameters')
                print(sess.run(all_trainable_vars))
            ep = ep + 1
            avg_cost = 0
            tic = time.time()
            try:
                for jj in range(total_batch):
                    # print('batch: ', jj)
                    tmp, _, _, output = sess.run([loss, update_ops, train_op, out])
                    # put TL here
                    avg_cost += tmp / total_batch
                    if (ii == 0 and jj == 0):
                        print('Iter: ', ii, 'Loss : ', tmp)
                # if (np.mod(ii,5)==0):
                #    sio.savemat(('Epoch'+str(ii)+'_Outputs.mat'),  {'input': r2c(np.squeeze(sess.run(atbT))), 'output': r2c(np.squeeze(output)),'all_intermediate_results': sess.run(all_intermediate_results)})
                # toc = time.time() - tic
                # totalLoss.append(avg_cost)
                # totalTime.append(toc)
                # sio.savemat((directory + '/TrainingLog2.mat'), {'loss': totalLoss})  # save the results
                # print("Epoch:", ii, "elapsed_time =""{:f}".format(toc), "cost =", "{:.3f}".format(avg_cost))

                toc = time.time() - tic
                totalLoss.append(avg_cost)
                totalTime.append(toc)
                print("Epoch:", ii, "elapsed_time =""{:f}".format(toc), "cost =", "{:.3f}".format(avg_cost))

            except tf.errors.OutOfRangeError:
                pass
            # if (np.mod(ep, 10) == 0 or ep==75 or ep==85 or ep==95):
            saver.save(sess, sessFileName, global_step=ep)
            sio.savemat((directory + '/TrainingLog.mat'), {'loss': totalLoss})  # save the results
        saver.save(sess, sessFileName, global_step=ep)
        # savedfile=saver.save(sess, sessFileName,global_step=ep,write_meta_graph=True)
    end_time = time.time()
    sio.savemat((directory + '/TrainingLog.mat'), {'loss': totalLoss})  # save the results
    print ('Training completed in minutes ', ((end_time - start_time) / 60))
    print ('training completed at', datetime.now().strftime("%d-%b-%Y %I:%M %P"))
    print ('*************************************************')
