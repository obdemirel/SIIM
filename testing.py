import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py as h5
import time
import hdf5storage
from scipy import signal
from scipy.linalg import solve


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
acc_rate = 4
data_opt = 'DL_perf'
supervised = False #not a supervised learning thereore it's OFF
chkPntnum = 100 # this is the epoch number you want to test

espirit = True
unnorm = False

if data_opt == 'DL_perf':

    #First we need the saved model directory
    if(espirit):
        file_dir = "/home/daedalus1-raid1/omer-data/perfusion_github/savedModels/perf_ESPIRiT_4R_10K_15RB_100E_LR_3e4_Uniform_Random40Perc_5Reps"
    if (unnorm):
        file_dir = "/home/daedalus1-raid1/omer-data/perfusion_github/savedModels/perf_SSIM_4R_10K_15RB_100E_LR_3e4_Uniform_Random40Perc_5Reps"
    #where to save
    #save_dir = '/home/daedalus1-raid1/omer-data/perfusion_github/Results/' + 'Subject' + str(subject_num) + '_perf_' + '.mat'



    slice_size, nrow_GLOB, ncol_GLOB, ncoil_GLOB = 3, 164, 172, 34  #Slice Number, RO, PE, Channel

    ##################################################################################################################################


    maps_list = list()
    rssq_maps_list = list()
    kspace_list = list()
    padded_mask_list = list()
    unpadded_mask_list = list()
    warm_start_list = list()
    subjects =  [2]

    test_if_all = True # testing on all time-frames

    save_recon_only = True #if you make this True, it will only save the final recons
    save_all = False #if you make this True, it will save all intermediate steps


    for subject_num in subjects:

        save_dir = '/home/daedalus1-raid1/omer-data/perfusion_github/Results/' + 'Subject' + str(subject_num) + '_perf' + '.mat'
        # choose the k-space, maps and mask files like in training
        file_dirtest = "/home/daedalus1-raid1/omer-data/perfusion_github/database/subject_"  # choose the database path
        
        slices = [1]  # if you have multiple slab groups

        for slice_num in slices:
            print ('Subject Num: ', subject_num)
            kspace = hdf5storage.loadmat(file_dirtest + str(subject_num) + ".mat")['kspace_all']
            maps = hdf5storage.loadmat(file_dirtest + str(subject_num) + ".mat")['sense_maps_all']

            padded_mask = hdf5storage.loadmat(file_dirtest + str(subject_num) + ".mat")['mask_all']
            unpadded_mask = hdf5storage.loadmat(file_dirtest + str(subject_num) + ".mat")['mask_all']
            intensity = hdf5storage.loadmat(file_dirtest + str(subject_num) + ".mat")['sense_maps_intensity_all']

            if (test_if_all):
                kspace = np.transpose(np.copy(kspace[:, :, :, :]), axes=(3, 0, 1, 2))

                maps = np.tile(maps[:, :, :, :, np.newaxis], (1, 1, 1, 1, intensity.shape[3]))
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
                            maps[ii,kk,...] = tmp

                padded_mask = np.transpose(np.copy(padded_mask[:, :, :, :]), axes=(3, 0, 1, 2))
                unpadded_mask = np.transpose(np.copy(unpadded_mask[:, :, :, :]), axes=(3, 0, 1, 2))



            print('kspace: ', kspace.shape, ', maps: ', maps.shape)
            print('padded mask: ', padded_mask.shape, ', unpadded mask: ', unpadded_mask.shape)
            # kspace_list.append(kspace[])

            maps_list.append(maps)
            kspace_list.append(kspace)
            padded_mask_list.append(padded_mask[..., 0])
            unpadded_mask_list.append(unpadded_mask[..., 0])
            print('kspace: ', np.concatenate(np.asarray(kspace_list), axis=0).shape, ', maps: ',
                  np.concatenate(np.asarray(maps_list), axis=0).shape, ',  padded mask : ',
                  np.concatenate(np.asarray(padded_mask_list), axis=0).shape)

            kspace_test = np.concatenate(np.asarray(kspace_list), axis=0)
            testCsm = np.concatenate(np.asarray(maps_list), axis=0)
            padded_mask = np.concatenate(np.asarray(padded_mask_list), axis=0)
            unpadded_mask = np.concatenate(np.asarray(unpadded_mask_list), axis=0)

            
            #################################################################################################################################


#Some useful functions
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


def sense1(input_kspace, sens_maps):
    [m, n, nc] = np.shape(sens_maps)
    image_space = ifft(input_kspace, axes=(0, 1), norm=None, unitary_opt=True)
    Eh_op = np.conj(sens_maps) * image_space
    Eh_op = np.sum(Eh_op, axis=2)

    return Eh_op

#This is basic SENSE recon for sanity check
def SENSE(kspace, maps):
    kin = kspace[:, 0:172:4, :]
    kin = ifft(kin)
    DATA2 = np.roll(kin, 22, axis=1)
    # DATA2 = kin

    sm1 = maps[0, ...]
    sm2 = maps[1, ...]
    sm3 = maps[2, ...]

    ksb = 43  # kspace.shape[1]/4

    sens_all = np.zeros((34, 12), dtype=complex)
    res = np.zeros((164, 172, 3), dtype=complex)

    # print('ksb',ksb)
    for row in range(DATA2.shape[0]):
        for col in range(DATA2.shape[1]):
            ms = np.block(DATA2[row, col, :])
            # print('ms', ms.shape)

            s11 = sm1[row, col, :]
            s12 = sm1[row, col + ksb, :]
            s13 = sm1[row, col + ksb + ksb, :]
            s14 = sm1[row, col + ksb + ksb + ksb, :]
            # print('s11', s11.shape)
            sens1 = (np.stack([s11, s12, s13, s14], axis=0)).transpose()

            s11 = sm2[row, col, :]
            s12 = sm2[row, col + ksb, :]
            s13 = sm2[row, col + ksb + ksb, :]
            s14 = sm2[row, col + ksb + ksb + ksb, :]
            sens2 = (np.stack([s11, s12, s13, s14], axis=0)).transpose()

            s11 = sm3[row, col, :]
            s12 = sm3[row, col + ksb, :]
            s13 = sm3[row, col + ksb + ksb, :]
            s14 = sm3[row, col + ksb + ksb + ksb, :]
            sens3 = (np.stack([s11, s12, s13, s14], axis=0)).transpose()

            # print('sens1', sens1.shape)
            # print('sens2', sens2.shape)
            # print('sens3', sens3.shape)
            # sens_all = np.stack([sens1,sens2,sens3],axis=0)
            sens_all[:, :4] = sens1
            sens_all[:, 4:8] = sens2
            sens_all[:, 8:12] = sens3

            # print('sens_all', sens_all.shape)
            # mhat = np.linalg.solve(sens_all,ms)
            # mhat = np.matmul( np.linalg.inv(np.matmul(sens_all.transpose(),sens_all)) , (np.matmul(sens_all.transpose(),ms)))
            mhat = solve(np.matmul(sens_all.conj().transpose(), sens_all), np.matmul(sens_all.conj().transpose(), ms))

            # print('mhat', mhat.shape)
            res[row, col, 0] = mhat[0,]
            res[row, col + ksb, 0] = mhat[1,]
            res[row, col + ksb + ksb, 0] = mhat[2,]
            res[row, col + ksb + ksb + ksb, 0] = mhat[3,]

            res[row, col, 1] = mhat[4,]
            res[row, col + ksb, 1] = mhat[5,]
            res[row, col + ksb + ksb, 1] = mhat[6,]
            res[row, col + ksb + ksb + ksb, 1] = mhat[7,]

            res[row, col, 2] = mhat[8,]
            res[row, col + ksb, 2] = mhat[9,]
            res[row, col + ksb + ksb, 2] = mhat[10,]
            res[row, col + ksb + ksb + ksb, 2] = mhat[11,]

    return np.concatenate((res[..., 0], res[..., 1], res[..., 2]), axis=0)


#same as in training
def fixval(cc):
    im_space = ifft(cc, axes=(0, 1), norm=None, unitary_opt=True)
    rssq_im = np.sqrt(np.sum(np.square(np.abs(im_space)), axis=2))

    thkfix = np.max(rssq_im.flatten())  # * np.max(rssq_im.flatten())

    return thkfix

#same as in training
def Mcal(sens_maps, maskp, thk):
    [m, n, nc] = np.shape(sens_maps)
    M = np.empty((m, n), dtype=np.complex64)
    p1 = np.sum(np.abs(np.square(sens_maps)), axis=2)
    p2 = np.sum(np.sum(np.sum(maskp, axis=2), axis=1), axis=0) / (m * n * nc)

    M = (p1 * p2)
    return M

#some more useful functions
def myNMSE(ref, recon):
    """ This function calculates NMSE between the original and
    the reconstructed     images"""
    ref, recon = np.abs(ref), np.abs(recon)
    nrmse = np.linalg.norm(ref - recon) / np.linalg.norm(ref)
    nmse = nrmse ** 2
    return nmse


def mySSIM(space_ref, space_rec):
    space_ref = np.squeeze(space_ref)
    space_rec = np.squeeze(space_rec)
    space_ref = space_ref / np.amax(np.abs(space_ref))
    space_rec = space_rec / np.amax(np.abs(space_ref))
    data_range = np.amax(np.abs(space_ref)) - np.amin(np.abs(space_ref))
    return compare_ssim(space_rec, space_ref, data_range=data_range,
                        gaussian_weights=True,
                        use_sample_covariance=False)

def myPSNR(org, recon):
    """ This function calculates PSNR between the original and
    the reconstructed     images"""
    print('org size', org.size)
    print('org max', np.abs(org.max()))
    mse = np.sum(np.square(np.abs(org - recon))) / org.size
    psnr = 20 * np.log10(np.abs(org.max()) / (np.sqrt(mse) + 1e-10))
    return psnr


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


cwd = os.getcwd()
tf.reset_default_graph()
subDirectory = file_dir
# %% read multi-channel dataset
if data_opt == 'DL_perf':
    print('size of the test data: ', np.shape(kspace_test), ', size of the sensitivities: ', np.shape(testCsm))
else:
    disp('Incorrect data type')

slice_scalar = []
for ii in range(np.shape(kspace_test)[0]):
    temp = kspace_test[ii, :, :, :]
    #normalize the k-space
    slice_scalar.append(np.max(np.abs(temp[:])))
    kspace_test[ii, :, :, :] = temp / np.max(np.abs(temp[:]))
    for kk in range(slice_size):
        temp = testCsm[ii, kk, ...]
        testCsm[ii, kk, ...] = temp

print('testing')

nSlice, nrow, ncol, ncoil = kspace_test.shape
origMask, trnMask, valMask = np.empty((nSlice, nrow, ncol), dtype=np.complex64), np.empty((nSlice, nrow, ncol),
                                                                                          dtype=np.complex64), np.empty((nSlice, nrow, ncol), dtype=np.complex64)

origMask = np.complex64(np.copy(padded_mask))
sensitivities = np.copy(testCsm)

valMask = np.copy(origMask)
trnMask = np.copy(origMask)
#sio.savemat(('valmasks2.mat'), {'trnMask': trnMask, 'valMask': valMask})

print('shape of Mask  : ', np.shape(origMask))
add_mask = valMask + trnMask
diff_mask = origMask - add_mask
testOrg = np.empty((nSlice, nrow * slice_size, ncol), dtype=np.complex64)
testAtb = np.empty((nSlice, nrow * slice_size, ncol), dtype=np.complex64)
basic_sense = np.empty((nSlice, nrow * slice_size, ncol), dtype=np.complex64)
testAtb_Theta = np.empty(temp.shape, dtype=np.complex64)
ref_kspace = np.empty(kspace_test.shape, dtype=np.complex64)
testM = np.empty((nSlice, nrow * slice_size, ncol), dtype=np.float32)
testFixVal = np.empty(nSlice, dtype=np.float32)
testWarm = np.empty((nSlice, nrow * slice_size, ncol), dtype=np.complex64)

print('getting the refs and aliased sense1 images')
for ii in range(np.shape(testOrg)[0]):
    if np.mod(ii, 50) == 0:
        print('Iteration: ', ii)
    proc_mask = origMask[ii]
    proc_maskV = trnMask[ii]
    proc_mask = np.tile(proc_mask[:, :, np.newaxis], (1, 1, ncoil))
    proc_maskV = np.tile(proc_maskV[:, :, np.newaxis], (1, 1, ncoil))
    sub_kspace = kspace_test[ii] * proc_mask
    ref_kspace[ii] = kspace_test[ii] * proc_maskV
    sense_recon = lambda z: sense1(z, testCsm[ii, ...])
    for kk in range(slice_size):
        idx_start, idx_end = kk * nrow_GLOB, (kk + 1) * nrow_GLOB
        testOrg[ii, idx_start:idx_end, ...] = sense1(kspace_test[ii, ...], testCsm[ii, kk, ...])
        testAtb[ii, idx_start:idx_end, ...] = sense1(sub_kspace, testCsm[ii, kk, ...])

        testM[ii, idx_start:idx_end, ...] = Mcal(testCsm[ii, kk, ...], proc_mask, 1e-7)

    testWarm[ii, ...] = testAtb[ii, ...]


    testFixVal[ii] = fixval(kspace_test[ii, ...])


sio.savemat(('fix_vals' + str(acc_rate) + '.mat'), {'sense': testFixVal})
testCsm = np.transpose(testCsm, (0, 1, 4, 2, 3))
testOrgAll = np.copy(testOrg)
testAtbAll = np.copy(testAtb)
testCsmAll = np.copy(testCsm)
# testWarmAll = np.copy(testWarm)
testWarmAll = np.copy(testWarm)
testMAll = np.copy(testM)
testFVAll = np.copy(testFixVal)
all_recon_slices = list()
all_lams = list()
all_recon_slices_postproc = list()
all_recon_slices_final = list()
all_ref_slices = list()
all_input_slices = list()
all_cardiac_kspace = list()
all_ref_kspace = list()
all_input_kspace = list()
total_ssim = []
total_nmse = []
total_ssim_postproc = []
total_nmse_postproc = []
total_ssim_input = []
total_nmse_input = []
total_test_dur = []
ssdu_loss = []
all_slices_intermediate_outputs = []
all_slices_warms = []
all_resnets = []
dc_outputs = []
print ('Now loading the model ...')
load_model_dir = subDirectory  # complete path
tf.reset_default_graph()
# loadChkPoint=tf.train.latest_checkpoint(load_model_dir)
loadChkPoint = file_dir + '/model-' + str(chkPntnum)
# loadChkPoint = file_dir + '/model'#-' + str(chkPntnum)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    new_saver = tf.train.import_meta_graph(load_model_dir + '/modelTst.meta')
    new_saver.restore(sess, loadChkPoint)
    trainable_collection_test = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
    nontrainable_variables_test = [sess.run(v) for v in trainable_collection_test]
    # ..................................................................................................................
    graph = tf.get_default_graph()
    predT = graph.get_tensor_by_name('out:0')
    ul_output = graph.get_tensor_by_name('predTst:0')
    landa = graph.get_tensor_by_name('lam:0')
    x0_output = graph.get_tensor_by_name('x0:0')
    all_intermediate_outputs = graph.get_tensor_by_name('all_intermediate_outputs:0')
    # ...................................................................................................................
    maskT = graph.get_tensor_by_name('mask:0')
    if not supervised:
        maskV = graph.get_tensor_by_name('testmaskV:0')
    atbT = graph.get_tensor_by_name('atb:0')
    csmT = graph.get_tensor_by_name('csm:0')
    MsT = graph.get_tensor_by_name('mst:0')
    FVT = graph.get_tensor_by_name('fvt:0')
    WarmT = graph.get_tensor_by_name('warmt:0')
    wts = sess.run(tf.global_variables())
    for ii in range(np.shape(testOrg)[0]):  #

        testOrg = np.copy(testOrgAll[ii, :, :])[np.newaxis]
        testAtb = np.copy(testAtbAll[ii, :, :])[np.newaxis]
        testCsm = np.copy(testCsmAll[ii, :, :, :])[np.newaxis]
        testWarm = np.copy(testWarmAll[ii, :, :])[np.newaxis]
        testMask = np.copy(origMask[ii, :, :])[np.newaxis]
        trnMask2 = np.copy(trnMask[ii, :, :])[np.newaxis]
        testValMask = np.copy(valMask[ii, :, :])[np.newaxis]
        testM = np.copy(testMAll[ii, :, :])[np.newaxis]
        testFV = np.copy(testFVAll[ii])[np.newaxis]
        testOrg, testAtb = c2r(testOrg), c2r(testAtb)
        testM = c2r(testM)
        testWarm = c2r(testWarm)

        if supervised:
            dataDict = {atbT: testAtb, maskT: testMask, csmT: testCsm}
        else:
            # dataDict={atbT:testAtb,maskT:testMask,maskV:testValMask,csmT:testCsm}
            dataDict = {atbT: testAtb, maskT: trnMask2, maskV: testValMask, csmT: testCsm, MsT: testM, FVT: testFV,
                        WarmT: testWarm}
        # rec=sess.run(predT,feed_dict=dataDict)
        tic = time.time()
        # nw_out, ul_output, x0, all_intermediate_outputs, lam
        rec, ul_out, x0, all_intermediate_outputs_temp, lam = sess.run(
            [predT, ul_output, x0_output, all_intermediate_outputs, landa], feed_dict=dataDict)
        toc = time.time() - tic
        testOrg = r2c(testOrg.squeeze())
        testAtb = r2c(testAtb.squeeze())
        rec = r2c(rec.squeeze())
        x0 = r2c(x0.squeeze())

        factor = 1
        testOrg = (testOrg) / factor
        # testAtb=np.abs(testAtb)/ factor
        testAtb = testAtb / factor
        # rec = np.abs(rec) / factor
        rec = rec / factor
        # ...............................................................................................................
        print('elapsed time %f seconds' % toc)
        vmin = np.min(np.abs(testOrg[:]))
        vmax_ref = np.max(np.abs(testOrg[:]))
        dc_outputs.append(x0)
        # all_slices_intermediate_outputs.append(np.abs(np.asarray(all_intermediate_outputs_temp)))
        all_slices_intermediate_outputs.append((np.asarray(all_intermediate_outputs_temp)))
        all_lams.append(lam)
        # all_ref_kspace.append(cardiac_ref_kspace)
        # all_input_kspace.append(cardiac_input_kspace)
        all_recon_slices.append(rec)
        all_ref_slices.append(testOrg)
        all_input_slices.append(testAtb)
        all_resnets.append(np.asarray(nontrainable_variables_test))
        # all_recon_slices_postproc.append(rec_postproc)
        total_test_dur.append(toc)


        final_recon_all = np.zeros(((3*(nrow + 44), ncol + 20)), dtype=np.complex64)
        for abc in range(3):

            final_recon = np.zeros(((nrow + 44, ncol + 20)), dtype=np.complex64)
            if (unnorm):
                # multiply with the intensities
                rec[abc * 164:(abc + 1) * 164] = rec[abc * 164:(abc + 1) * 164]*intensity[ii,abc,:,:,0]
            final_recon[44:, :ncol] = np.roll(fft(np.asarray(rec[abc * 164:(abc + 1) * 164]), axes=(0, 1), norm=None, unitary_opt=True), (10,-22), axis=(1,0))
            final_recon = ifft(final_recon, axes=(0, 1), norm=None, unitary_opt=True)
            final_recon_all[abc * 208:(abc + 1) * 208,:] = final_recon
        all_recon_slices_final.append(final_recon_all * slice_scalar[ii])

        print('ITERATION --------------->', ii)


if save_recon_only:
    sio.savemat((save_dir), {'final_recon': all_recon_slices_final})
if save_all:
    sio.savemat((save_dir),
                {'ref_slices': all_ref_slices, 'lams': all_lams, 'final_recon': all_recon_slices_final, 'input': all_input_slices,
                 'all_intermediate_outputs': all_slices_intermediate_outputs, 'cg_output': (np.asarray(dc_outputs))})
