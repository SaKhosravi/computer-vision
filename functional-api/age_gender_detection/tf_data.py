import tensorflow as tf
import pathlib
import numpy as np
import os

from matplotlib import pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE


class TF_Dataa:
    def __init__(self, directory,
                 validation_size=0.2,
                 test_size=0.3,

                 batch_size=32,
                 shuffle_buffer_size=1000,
                 augmentation=False,
                 shuffle_dataset=True,
                 input_target_size=[150, 150, 3]
                 ):
        self.directory = directory
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.augmentation = augmentation
        self.shuffle_dataset = shuffle_dataset
        self.validation_size = validation_size
        self.input_target_size = input_target_size
        self.test_size = test_size


    def augment_data(self, image):
        a = self.input_target_size[0] + 30
        b = self.input_target_size[1] + 30
        image = tf.image.resize_with_crop_or_pad(image, a, b)
        image = tf.image.random_brightness(image, max_delta=0.5)
        image = tf.image.random_flip_left_right(image, seed=123)
        return image

    def get_classNames(self, ):
        train_dir = pathlib.Path(self.directory)
        CLASSNAME = np.array([item.name for item in train_dir.glob('*') if item.name != "LICENSE.txt"])
        return CLASSNAME

    def get_fullDataset(self):
        print("********* get_fullDataset *************** ")

        full_data = tf.data.Dataset.list_files(str(self.directory + "*/*"))
        # if self.shuffle_dataset:
        #   full_data = tf.data.Dataset.shuffle(self, seed=100, buffer_size = AUTOTUNE)
        return full_data

    def getTrainValidationData(self, ):
        full_dataset = self.get_fullDataset()

        DATASET_SIZE = len(list(full_dataset))
        print("full datast size is: {} ".format(DATASET_SIZE))

        train_size = int((1 - (self.validation_size + self.test_size)) * DATASET_SIZE)
        test_size = int(self.test_size * DATASET_SIZE)
        val_size = int(self.validation_size * DATASET_SIZE)

        train_dataset = full_dataset.take(train_size)
        remaining = full_dataset.skip(train_size)

        test_dataset = remaining.take(test_size)
        val_dataset = remaining.skip(test_size)

        print("train size: {}, validation size:{} , test size ={}".format(
            len(list(train_dataset)), len(list(val_dataset)), len(list(test_dataset))
        ))

        return train_dataset, val_dataset, test_dataset

    def get_label(self, file_path):
        img_name = tf.strings.split(file_path, "UTKFace/")[1]
        age = tf.strings.to_number(
                tf.strings.split(img_name, "_")[0],
                out_type=tf.dtypes.int32,
                name=None
            )

        
        gender = tf.strings.to_number(
                tf.strings.split(img_name, "_")[1],
                out_type=tf.dtypes.int32,
                name=None
            )

        
        return {"gender" : gender,"age":age } 

    def load_image(self, image_path):
        print("********* getTrainValidationData *************** ", len(self.input_target_size))
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, 3, expand_animations=False)
        img = tf.cast(img, tf.float32)
        return img

    def normalize(self, image):
        image = (image / 127.5) - 1
        return image

    def resize_image(self, image, height, width):
        image = tf.image.resize(image,
                                (height, width),
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
                                )

        return image

    def load_image_with_label(self, image_path):
        print("********* load_image_with_label ***************")

        lable = self.get_label(image_path)
        image = self.load_image(image_path)


        return image, lable

    def load_image_train(self, image_path):
        print("********* load_image_train ***************")

        image, lable = self.load_image_with_label(image_path)
        if self.augmentation:
            image = self.augment_data(image)

        image = self.resize_image(image, self.input_target_size[0],
                                  self.input_target_size[1], )
        image = self.normalize(image)

        return image, lable

    def load_image_test(self, image_path):
        print("********* load_image_test ***************")

        image, label = self.load_image_with_label(image_path)
        image = self.resize_image(image, self.input_target_size[0],
                                  self.input_target_size[1], )
        image = self.normalize(image)
        return image, label

    def train_validation_split(self):
        print("********* train_validation_split ***************")
        train_dataset, validation_dataset, test_dataset = self.getTrainValidationData()

        train_dataset = train_dataset.map(self.load_image_train)
        train_dataset = train_dataset.shuffle(self.shuffle_buffer_size)
        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.prefetch(AUTOTUNE)

        validation_dataset = validation_dataset.map(self.load_image_test)
        validation_dataset = validation_dataset.batch(self.batch_size)
        validation_dataset = validation_dataset.prefetch(AUTOTUNE)

        test_dataset = test_dataset.map(self.load_image_test)
        test_dataset = test_dataset.batch(self.batch_size)
        test_dataset = test_dataset.prefetch(AUTOTUNE)

        return train_dataset, validation_dataset, test_dataset




