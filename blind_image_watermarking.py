# Qiyuan Gong
# qiyuangong@gmail.com
# 2016-35-10

# numpy is needed
import pdb
import numpy as np
# pip install PyWavelets
# import pywt
# from matplotlib import pyplot
# from scipy.misc import imread
from skimage.io import imread
from skimage import img_as_ubyte, exposure
from matplotlib import pyplot as plt
from matplotlib.image import imsave
import pickle


def encode_watermark(original, watermark, alpha=1):
    """
    Add watermark to target image.
    """
    im = imread(original).astype('float') / 255
    mark = imread(watermark).astype('float') / 255
    imsize = im.shape
    TH = np.zeros((imsize[0] / 2, imsize[1], imsize[2]))
    TH1 = TH.copy()
    TH1[:mark.shape[0], :mark.shape[1], :] = mark
    # generate permuation from 0~n-1
    M = np.random.permutation(imsize[0] / 2)
    N = np.random.permutation(imsize[1])
    with open('encode.pickle', 'wb') as tmp_file:
        pickle.dump((M, N), tmp_file)
    for i in range(imsize[0] / 2):
        for j in range(imsize[1]):
            TH[i, j, :] = TH1[M[i], N[j], :]
    # print TH
    mark_t = np.zeros(imsize)
    mark_t[:imsize[0] / 2, :imsize[1], :] = TH
    for i in range(imsize[0] / 2):
        for j in range(imsize[1]):
                    mark_t[imsize[0] - i - 1, imsize[1] - j - 1, :] = TH[i, j, :]
    FA = np.fft.fft2(im, axes=(0, 1))
    FB = FA + alpha * mark_t
    FAO = np.fft.ifft2(FB, axes=(0, 1)).astype('float')
    # if saved to jpg, you will lost the watermark
    # watermarked_image = plt.figure('watermarked_image')
    # plt.imshow(FAO)
    # watermarked_image.show()
    # pdb.set_trace()
    # imsave('watermarked_img.jpg', FAO)
    # watermarked_image.savefig('watermarked_img.jpg')
    with open(original.split('.')[0] + '_raw', 'wb') as w_raw_file:
        pickle.dump(FAO, w_raw_file)


def decode_watermark(original, watermarked_file, alpha=1):
    """
    Detect and extract watermark from image.
    """
    im = imread(original).astype('float') / 255
    with open(watermarked_file, 'rb') as w_raw_file:
        FAO = pickle.load(w_raw_file)
    FA = np.fft.fft2(im, axes=(0, 1)).astype('float')
    FA2 = np.fft.fft2(FAO, axes=(0, 1)).astype('float')
    imsize = im.shape
    G = (FA2 - FA) / alpha
    GG = G.copy()
    with open('encode.pickle', 'rb') as tmp_file:
        M, N = pickle.load(tmp_file)
    for i in range(imsize[0] / 2):
        for j in range(imsize[1]):
            GG[M[i], N[j], :] = G[i, j, :]
    for i in range(imsize[0] / 2):
        for j in range(imsize[1]):
            GG[imsize[0] - 1 - i, imsize[1] - 1 - j, :] = GG[i, j, :]
    imsave('extracted_watermark.jpg', GG)
    # extracted_watermark = plt.figure('extracted_watermark')
    # plt.imshow(GG)
    # extracted_watermark.show()
    # pdb.set_trace()


if __name__ == '__main__':
    encode_watermark('gl1.jpg', 'watermark.jpg')
    decode_watermark('gl1.jpg', 'gl1_raw')
