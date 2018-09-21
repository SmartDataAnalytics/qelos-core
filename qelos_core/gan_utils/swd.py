# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import scipy.ndimage
import torch


# pytorch
def batch_to_patches(x, patch_size, patches_per_image):
    """
    :param x:                   torch tensor with images in batch (batsize, numchannels, height, width)
    :param patch_size:          size of patch
    :param patches_per_image:
    :return:
    """
    device = x.device
    assert(x.dim() == 4)
    batsize, nchannels, height, width = x.size()        # (minibatch, channel, height, width)
    assert(nchannels == 3)
    patch_count = patches_per_image * batsize
    hs = patch_size // 2
    patch_id, chan, x, y = np.ogrid[0:batsize, 0:3, -hs:hs+1, -hs:hs+1]
    patch_id, chan, x, y = [torch.tensor(_e, device=device) for _e in (patch_id, chan, x, y)]
    img_id = patch_id // patches_per_image
    x = x + torch.randint(hs, width -hs, size=(patch_count, 1, 1, 1), device=device, dtype=torch.int64)
    y = y + torch.randint(hs, height-hs, size=(patch_count, 1, 1, 1), device=device, dtype=torch.int64)
    idx = ((img_id * nchannels + chan) * height + y) * width + x
    patches = x.view(-1)[idx]
    return patches      # (patch_count, nchannels, patch_size, patch_size)


def finalize_patches(patches):
    """
    Concatenate, normalize and reshape patches
    :param patches: list of patches of size (patch_count, nchannels, patch_size, patch_size)
    :return:
    """
    if isinstance(patches, list):
        patches = torch.cat(patches, axis=0)
    assert(patches.dim() == 4)      # (patch_count_for_all_batches, nchannels, patch_size, patch_size)
    _patches = patches.transpose(0, 1).contiguous().view(patches.size(1), -1)
    patches -= _patches.mean(1).view(1, -1, 1, 1)
    patches /= _patches.std(1, unbiased=True).view(1, -1, 1, 1)
    patches = patches.view(patches.size(0), -1)
    return patches


def sliced_wasserstein_(A, B, dir_repeats, dirs_per_repeat):
    assert(A.dim() == 2 and A.size() == B.size())
    results = []
    for repeat in range(dir_repeats):
        dirs = torch.randn(A.size(1), dirs_per_repeat, device=A.device)
        dirs /= (dirs ** 2).sum(0, True).sqrt()
        projA = torch.matmul(A, dirs)
        projB = torch.matmul(B, dirs)
        projA = projA.sort(0)[0]
        projB = projB.sort(0)[0]
        dists = (projA - projB).abs()
        results.append(dists.mean().item())
    return np.mean(results)




# numpy
def get_descriptors_for_minibatch(minibatch, nhood_size, nhoods_per_image):
    S = minibatch.shape # (minibatch, channel, height, width)
    assert len(S) == 4 and S[1] == 3
    N = nhoods_per_image * S[0]
    H = nhood_size // 2
    nhood, chan, x, y = np.ogrid[0:N, 0:3, -H:H+1, -H:H+1]
    img = nhood // nhoods_per_image
    x = x + np.random.randint(H, S[3] - H, size=(N, 1, 1, 1))
    y = y + np.random.randint(H, S[2] - H, size=(N, 1, 1, 1))
    idx = ((img * S[1] + chan) * S[2] + y) * S[3] + x
    return minibatch.flat[idx]


def finalize_descriptors(desc):
    if isinstance(desc, list):
        desc = np.concatenate(desc, axis=0)
    assert desc.ndim == 4 # (neighborhood, channel, height, width)
    desc -= np.mean(desc, axis=(0, 2, 3), keepdims=True)
    desc /= np.std(desc, axis=(0, 2, 3), keepdims=True)
    desc = desc.reshape(desc.shape[0], -1)
    return desc


def sliced_wasserstein(A, B, dir_repeats, dirs_per_repeat):
    assert A.ndim == 2 and A.shape == B.shape                           # (neighborhood, descriptor_component)
    results = []
    for repeat in range(dir_repeats):
        dirs = np.random.randn(A.shape[1], dirs_per_repeat)             # (descriptor_component, direction)
        dirs /= np.sqrt(np.sum(np.square(dirs), axis=0, keepdims=True)) # normalize descriptor components for each direction
        dirs = dirs.astype(np.float32)
        projA = np.matmul(A, dirs)                                      # (neighborhood, direction)
        projB = np.matmul(B, dirs)
        projA = np.sort(projA, axis=0)                                  # sort neighborhood projections for each direction
        projB = np.sort(projB, axis=0)
        dists = np.abs(projA - projB)                                   # pointwise wasserstein distances
        results.append(np.mean(dists))                                  # average over neighborhoods and directions
    return np.mean(results)                                             # average over repeats


def downscale_minibatch(minibatch, lod):
    if lod == 0:
        return minibatch
    t = minibatch.astype(np.float32)
    for i in range(lod):
        t = (t[:, :, 0::2, 0::2] + t[:, :, 0::2, 1::2] + t[:, :, 1::2, 0::2] + t[:, :, 1::2, 1::2]) * 0.25
    return np.round(t).clip(0, 255).astype(np.uint8)


gaussian_filter = np.float32([
    [1, 4,  6,  4,  1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4,  6,  4,  1]]) / 256.0


def pyr_down(minibatch): # matches cv2.pyrDown()
    assert minibatch.ndim == 4
    return scipy.ndimage.convolve(minibatch, gaussian_filter[np.newaxis, np.newaxis, :, :], mode='mirror')[:, :, ::2, ::2]


def pyr_up(minibatch): # matches cv2.pyrUp()
    assert minibatch.ndim == 4
    S = minibatch.shape
    res = np.zeros((S[0], S[1], S[2] * 2, S[3] * 2), minibatch.dtype)
    res[:, :, ::2, ::2] = minibatch
    return scipy.ndimage.convolve(res, gaussian_filter[np.newaxis, np.newaxis, :, :] * 4.0, mode='mirror')


def generate_laplacian_pyramid(minibatch, num_levels):
    pyramid = [np.float32(minibatch)]
    for i in range(1, num_levels):
        pyramid.append(pyr_down(pyramid[-1]))
        pyramid[-2] -= pyr_up(pyramid[-1])
    return pyramid


def reconstruct_laplacian_pyramid(pyramid):
    minibatch = pyramid[-1]
    for level in pyramid[-2::-1]:
        minibatch = pyr_up(minibatch) + level
    return minibatch


class SlicedWassersteinDistance(object):
    def __init__(self, image_shape, nhood_size=7, nhoods_per_image=128, dir_repeats=4, dirs_per_repeat=128):
        self.nhood_size         = nhood_size
        self.nhoods_per_image   = nhoods_per_image
        self.dir_repeats        = dir_repeats
        self.dirs_per_repeat    = dirs_per_repeat
        self.resolutions = []
        res = image_shape[1]
        while res >= 16:
            self.resolutions.append(res)
            res //= 2

    def get_metric_names(self):
        return ['SWDx1e3_%d' % res for res in self.resolutions] + ['SWDx1e3_avg']

    def get_metric_formatting(self):
        return ['%-13.4f'] * len(self.get_metric_names())

    def begin(self, mode):
        assert mode in ['warmup', 'reals', 'fakes']
        self.descriptors = [[] for res in self.resolutions]

    def feed(self, mode, minibatch):
        for lod, level in enumerate(generate_laplacian_pyramid(minibatch, len(self.resolutions))):
            desc = get_descriptors_for_minibatch(level, self.nhood_size, self.nhoods_per_image)
            self.descriptors[lod].append(desc)

    def end(self, mode):
        desc = [finalize_descriptors(d) for d in self.descriptors]
        del self.descriptors
        if mode in ['warmup', 'reals']:
            self.desc_real = desc
        dist = [sliced_wasserstein(dreal, dfake, self.dir_repeats, self.dirs_per_repeat) for dreal, dfake in
                zip(self.desc_real, desc)]
        del desc
        dist = [d * 1e3 for d in dist] # multiply by 10^3
        return dist + [np.mean(dist)]

#----------------------------------------------------------------------------

def tst_numpy():
    images = np.random.random((1000, 3, 32, 32))
    image_batches = np.split(images, 20)
    print(image_batches[0].shape)
    print(len(image_batches))
    real_batches = image_batches[:10]
    fake_batches = image_batches[10:]
    swd = SlicedWassersteinDistance((3, 32, 32))

    swd.begin("reals")
    for batch in real_batches:
        swd.feed("reals", batch)
    dist = swd.end("reals")
    print(dist)

    swd.begin("fakes")
    for batch in fake_batches:
        swd.feed("fakes", batch)
    dist = swd.end("fakes")
    print(dist)



if __name__ == '__main__':
    tst_numpy()