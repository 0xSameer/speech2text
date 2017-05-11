import gzip
import os
import struct

import numpy
import six

from chainer.dataset import download
from chainer.datasets import tuple_dataset

from basics import *
from nn_config import *

def get_callhome():
    """Get the callhome data.

    Returns:
        A tuple of two datasets. both datasets
        are :class:`~chainer.datasets.TupleDataset` instances.

    """
    train_raw = _retrieve_mnist_training()
    train = _preprocess_mnist(train_raw, withlabel, ndim, scale, dtype,
                              label_dtype)
    test_raw = _retrieve_mnist_test()
    test = _preprocess_mnist(test_raw, withlabel, ndim, scale, dtype,
                             label_dtype)
    return train, test


def _preprocess_mnist(raw, withlabel, ndim, scale, image_dtype, label_dtype):
    images = raw['x']
    labels = raw['y'].astype(label_dtype)
    return tuple_dataset.TupleDataset(images, labels)


def _retrieve_mnist_training():
    return _retrieve_mnist('train.npz', urls)


def _retrieve_mnist_test():
    return _retrieve_mnist('dev.npz', urls)


def _retrieve_mnist(name, urls):
    path = os.path.join(input_dir, name)

    if not os.path.exists(path):
        make_npz(path, name.replace(".npz", ""))

    return np.load(path)


def make_npz(path, set_type):
    x_path = speech_dir
    # y_path = text_dir

    frame_lengths = []

    text_dict = pickle.load(open(text_data_dict, "rb"))

    set_files = sorted([f for f in text_dict[set_type].keys()])

    sp_files = ["{0:s}{1:s}".format(os.path.join(speech_dir, f), speech_extn) for f in set_files]

    N = len(set_files)
    x = np.empty((N,), dtype=np.float32)
    y_char = np.empty((N,), dtype=np.int32)
    y_word = np.empty((N,), dtype=np.int32)

    for i, sp_fil in tqdm(enumerate(sp_files)):
        #print(sp_fil)
        temp_arr = np.load(sp_fil)
        frame_lengths.append(temp_arr.shape[0])

    print(np.mean(frame_lengths), np.std(frame_lengths), np.max(frame_lengths))

    return frame_lengths


    # with gzip.open(x_path, 'rb') as fx, gzip.open(y_path, 'rb') as fy:
    #     fx.read(4)
    #     fy.read(4)
    #     N, = struct.unpack('>i', fx.read(4))
    #     if N != struct.unpack('>i', fy.read(4))[0]:
    #         raise RuntimeError('wrong pair of MNIST images and labels')
    #     fx.read(8)

    #     x = numpy.empty((N, 784), dtype=numpy.uint8)
    #     y = numpy.empty(N, dtype=numpy.uint8)

    #     for i in six.moves.range(N):
    #         y[i] = ord(fy.read(1))
    #         for j in six.moves.range(784):
    #             x[i, j] = ord(fx.read(1))

    # numpy.savez_compressed(path, x=x, y=y)
    # return {'x': x, 'y': y}


make_npz("haha", "train")