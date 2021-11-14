#
# WARNING: highly unefficient, with code redundancy. It's only to get the data once for a homework.
#

import sys

from tarea1.mapping import animals

sys.path.append("/content/convnet2")
import tensorflow as tf
import models.resnet as resnet
import datasets.data as data
import utils.configuration as conf
import utils.imgproc as imgproc
import skimage.io as io
import skimage.transform as trans
import os
import argparse
import numpy as np
import statistics


class SSearch:
    def __init__(self, config_file, model_name):

        self.configuration = conf.ConfigurationFile(config_file, model_name)
        # loading data
        mean_file = os.path.join(self.configuration.get_data_dir(), "mean.dat")
        shape_file = os.path.join(self.configuration.get_data_dir(), "shape.dat")
        #
        self.input_shape = np.fromfile(shape_file, dtype=np.int32)
        self.mean_image = np.fromfile(mean_file, dtype=np.float32)
        self.mean_image = np.reshape(self.mean_image, self.input_shape)

        # loading classifier model
        model = resnet.ResNet([3, 4, 6, 3], [64, 128, 256, 512], self.configuration.get_number_of_classes(),
                              se_factor=0)
        input_image = tf.keras.Input((self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                                     name='input_image')
        model(input_image)
        model.summary()
        model.load_weights(self.configuration.get_checkpoint_file(), by_name=True, skip_mismatch=True)
        self.sim_model = model
        print('sim_model was loaded OK')
        # defining process arch
        self.process_fun = imgproc.process_sketch
        # loading catalog
        self.ssearch_dir = os.path.join(self.configuration.get_data_dir(), 'ssearch')
        catalog_file = os.path.join(self.ssearch_dir, 'catalog.txt')
        assert os.path.exists(catalog_file), '{} does not exist'.format(catalog_file)
        print('loading catalog ...')
        self.load_catalog(catalog_file)
        print('loading catalog ok ...')
        self.enable_search = False

        # load model

    def read_image(self, filename):
        target_size = (self.configuration.get_image_height(), self.configuration.get_image_width())
        image = self.process_fun(data.read_image(filename, self.configuration.get_number_of_channels()), target_size)
        return image

    def load_features(self):
        fvs_file = os.path.join(self.ssearch_dir, "features.np")
        fshape_file = os.path.join(self.ssearch_dir, "features_shape.np")
        features_shape = np.fromfile(fshape_file, dtype=np.int32)
        self.features = np.fromfile(fvs_file, dtype=np.float32)
        self.features = np.reshape(self.features, features_shape)
        self.enable_search = True
        print('features loaded ok')

    def load_catalog(self, catalog):
        with open(catalog) as f_in:
            self.filenames = [filename.strip() for filename in f_in]
        self.data_size = len(self.filenames)

    def get_filenames(self, idxs):
        return [self.filenames[i] for i in idxs]

    def compute_features(self, image, expand_dims=False):
        image = image - self.mean_image
        if expand_dims:
            image = tf.expand_dims(image, 0)
        fv = self.sim_model.predict(image)
        return fv

    def normalize(self, data):
        norm = np.sqrt(np.sum(np.square(data), axis=1))
        norm = np.expand_dims(norm, 0)
        data = data / np.transpose(norm)
        return data

    def search(self, im_query):
        assert self.enable_search, 'search is not allowed'
        q_fv = self.compute_features(im_query, expand_dims=True)
        sim = np.matmul(self.normalize(self.features), np.transpose(self.normalize(q_fv)))
        sim = np.reshape(sim, (-1))
        # it seems that Euclidean performs better than cosine
        # d = np.sqrt(np.sum(np.square(self.features - q_fv[0]), axis=1))
        # This is cosine distance
        idx_sorted = np.argsort(-sim)
        # idx_sorted = np.argsort(d)
        return idx_sorted[1:]

    def compute_features_from_catalog(self):
        n_batch = self.configuration.get_batch_size()
        images = np.empty((self.data_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                          dtype=np.uint8)
        for i, filename in enumerate(self.filenames):
            if i % 1000 == 0:
                print('reading {}'.format(i))
                sys.stdout.flush()
            images[i,] = self.read_image(filename)
        n_iter = np.int(np.ceil(self.data_size / n_batch))
        result = []
        for i in range(n_iter):
            print('iter {} / {}'.format(i, n_iter))
            sys.stdout.flush()
            batch = images[i * n_batch: min((i + 1) * n_batch, self.data_size), ]
            result.append(self.compute_features(batch))
        fvs = np.concatenate(result)
        print('fvs {}'.format(fvs.shape))
        fvs_file = os.path.join(self.ssearch_dir, "features.np")
        fshape_file = os.path.join(self.ssearch_dir, "features_shape.np")
        np.asarray(fvs.shape).astype(np.int32).tofile(fshape_file)
        fvs.astype(np.float32).tofile(fvs_file)
        print('fvs saved at {}'.format(fvs_file))
        print('fshape saved at {}'.format(fshape_file))

    def draw_result(self, filenames):
        w = 1000
        h = 1000
        w_i = np.int(w / 10)
        h_i = np.int(h / 10)
        image_r = np.zeros((w, h, 3), dtype=np.uint8) + 255
        x = 0
        y = 0
        for i, filename in enumerate(filenames):
            pos = (i * w_i)
            x = pos % w
            y = np.int(np.floor(pos / w)) * h_i
            image = self.read_image(filename)
            image = imgproc.toUINT8(trans.resize(image, (h_i, w_i)))
            image_r[y:y + h_i, x: x + w_i, :] = image
        return image_r

    # unit test


def get_animal(path_str):
    if not path_str:
        sys.exit('empty string')
    if "/" not in path_str:
        sys.exit('invalid string')
    count = path_str.count("/")
    if count < 2:
        sys.exit('invalid string 2')
    splitted = path_str.split("/")
    return splitted[len(splitted) - 2].strip().lower()


def AP(ap_dict, fquery):
    im_query = ssearch.read_image(fquery)
    idx = ssearch.search(im_query)
    animal = get_animal(fquery)
    r_filenames = ssearch.get_filenames(idx)
    r_filenames.insert(0, fquery)
    total = 0
    n_relevants = 0
    # print(r_filenames[:10])
    # print(f"'{animal}'")
    animal2 = get_animal(r_filenames[1])
    total = total + 1
    ap = 0.0
    if animal == animal2:
        n_relevants = 1
        ap = 100.0
    for ix, filepath in enumerate(r_filenames):
        if ix < 2:
            continue
        animal2 = get_animal(filepath)
        # print(f"ix={ix},ap={ap}, file path='{filepath}', matches? {animal == animal2}")
        total = total + 1
        if animal == animal2:
            n_relevants = n_relevants + 1
            new_p = 100.0 * n_relevants / ix
            ap = (ap * (n_relevants - 1) + new_p) / n_relevants
    # print(f"'{animal}',n_relevants={n_relevants},total={total},gross precision={100.0 * n_relevants / total},ap={ap}")
    return ap


def precision_at_1(p1_dict, fquery):
    im_query = ssearch.read_image(fquery)
    idx = ssearch.search(im_query)
    animal = get_animal(fquery)
    r_filenames = ssearch.get_filenames(idx)
    r_filenames.insert(0, fquery)
    total = 0
    animal2 = get_animal(r_filenames[1])
    total = total + 1
    if animal == animal2:
        p1_dict['total'].append(100.0)
        p1_dict[animal].append(100.0)
        return
    for count, filepath in enumerate(r_filenames):
        if count < 2:
            continue
        animal2 = get_animal(filepath)
        total = total + 1
        if animal == animal2:
            p1_dict['total'].append(100.0 * 1 / count)
            p1_dict[animal].append(100.0 * 1 / count)
            return
    p1_dict['total'].append(0.0)
    p1_dict[animal].append(0.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Similarity Search")
    parser.add_argument("-config", type=str, help="<str> configuration file", required=True)
    parser.add_argument("-name", type=str, help=" name of section in the configuration file", required=True)
    parser.add_argument("-mode", type=str, choices=['search', 'compute', 'tarea1'], help=" mode of operation",
                        required=True)
    parser.add_argument("-list", type=str, help=" list of image to process", required=False)
    parser.add_argument("-odir", type=str, help=" output dir", required=False, default='.')
    pargs = parser.parse_args()
    configuration_file = pargs.config
    ssearch = SSearch(pargs.config, pargs.name)
    # filename = '/home/vision/smb-datasets/clothing-dataset/classifier_data/dress/91f98166-e897-47d8-b.png'
    # fv = ssearch.compute_features(filename)
    if pargs.mode == 'compute':
        ssearch.compute_features_from_catalog()

    if pargs.mode == 'search':
        ssearch.load_features()
        # fquery =  '/home/vision/smb-datasets/clothing-dataset/classifier_data/dress/1b550b68-e499-49dd-9.png'
        # fquery = '/home/vision/smb-datasets/missodd/queries/missodd-query-2.png'
        if pargs.list is not None:
            with open(pargs.list) as f_list:
                filenames = [item.strip() for item in f_list]
            for fquery in filenames:
                im_query = ssearch.read_image(fquery)
                idx = ssearch.search(im_query)
                r_filenames = ssearch.get_filenames(idx)
                r_filenames.insert(0, fquery)  #
                image_r = ssearch.draw_result(r_filenames)
                output_name = os.path.basename(fquery) + '_result.png'
                output_name = os.path.join(pargs.odir, output_name)
                io.imsave(output_name, image_r)
                print('result saved at {}'.format(output_name))
        else:
            fquery = input('Query:')
            while fquery != 'quit':
                im_query = ssearch.read_image(fquery)
                idx = ssearch.search(im_query)
                # print(idx)
                r_filenames = ssearch.get_filenames(idx)
                r_filenames.insert(0, fquery)
                #             for f in r_filenames :
                #                 print(f)
                image_r = ssearch.draw_result(r_filenames)
                output_name = os.path.basename(fquery) + '_result.png'
                output_name = os.path.join(pargs.odir, output_name)
                io.imsave(output_name, image_r)
                print('result saved at {}'.format(output_name))
                fquery = input('Query:')

    if pargs.mode == 'tarea1':
        ssearch.load_features()
        ap_dict = {'total': []}
        p1_dict = {'total': []}
        for (i, animal) in enumerate(animals):
            ap_dict[animal] = []
            p1_dict[animal] = []

        file1 = open('/content/convnet2/data/sketch_folder/ssearch/catalog.txt', 'r')
        Lines = file1.readlines()
        counter = 0
        for line in Lines:
            fpath = line.strip()
            print(f"{counter}")
            # print(f"{counter},{get_animal(fpath)},{fpath},{AP(fpath)},{precision_at_1(fpath)}")
            # AP(ap_dict, fpath)
            precision_at_1(p1_dict, fpath)
            counter = counter + 1

        print(f"THIS IS IMPORTANT")
        for animal, arr in p1_dict.items():
            print(f"{animal},{statistics.mean(arr)}")
        print(f"END IMPORTANT")
