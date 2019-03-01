import os
import cv2
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.misc as ms


class DataGenerator:
    def __init__(self, configs):
        self.image_dir = configs.image_dir
        self.scale = configs.scale

        if os.path.isfile(configs.preprocessed_data):
            print('Data file found!..............')
            with open(configs.preprocessed_data, 'rb') as input_file:
                merged_data = pickle.load(input_file)
                self.images, self.labels = zip(*merged_data)
                self.images = np.asarray(self.images)
                self.labels = np.asarray(self.labels)
        else:
            self.images, self.labels = self.prep_images(self.image_dir, 33, 21, 20)
            print('Saving data as picklei............')
            self.save_as(configs.preprocessed_data)
            print('Data saved successfully......')


    def modcrop(self, image, scale):
        size = image.shape
        size = size - np.mod(size, scale)
        image = image[:size[0], :size[1]]
        return image

    def prep_images(self, image_dir, input_shape, label_shape, stride):
        data  = np.zeros((1, input_shape, input_shape, 3))
        label = np.zeros((1, label_shape, label_shape, 3))


        padding = (input_shape - label_shape)//2
        #print(type(self.scale))
        for file in os.listdir(image_dir):
            for image in os.listdir(image_dir + '/' + file):
                if(image != 'desktop.ini'):
                    img = cv2.imread(image_dir+ '/' +file + '/' + image)
                    img_label = self.modcrop(img, self.scale)
                    sz = img_label.shape
                    img_input = ms.imresize(ms.imresize(img_label, (sz[0]//self.scale, sz[1]//self.scale), interp='bicubic'), sz, interp='bicubic')
                    #print(sz, ' ', im_input.shape)


                    for x in np.arange(0, sz[1]-input_shape, stride):
                        for y in np.arange(0, sz[0]-input_shape, stride):
                            subimg_inp = img_input[y:y+input_shape, x:x+input_shape]
                            subimg_label = img_label[y+padding: y+padding+label_shape, x+padding: x+padding+label_shape]

                            subimg_inp = np.expand_dims(subimg_inp, axis=0)
                            subimg_label = np.expand_dims(subimg_label, axis=0)
                            #print(data.shape)
                            #print(subimg_inp.shape)
                            data = np.vstack((data, subimg_inp))
                            label = np.vstack((label, subimg_label))
            break
        return data, label

    def get_next_batch(self, batch_size=10):
        data_len = self.images.shape[0]
        for i in np.arange(1, data_len, batch_size):
            yield(self.images[i:i+batch_size], self.labels[i:i+batch_size])

    def split_data(self, data):
        x_train, x_test = train_test_split(np.array(data), test_size=0.2)
        return x_train, x_test

    def save_as(self, filename):
        merged_data = list(zip(self.images, self.labels))
        with open(filename, 'wb') as pickle_file:
            pickle.dump(merged_data, pickle_file)

