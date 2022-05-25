import tensorflow as tf
import pathlib
import numpy as  np
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE


class TF_Dataa:
    def __init__(self, directory,
                 batch_size=32,
                 shuffle_buffer_size=1000,
                 augmentation=False,
                 validation_size=0.2,
                 input_target_size=[150, 150, 3]
                 ):
        self.directory = directory
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.augmentation = augmentation
        self.validation_size = validation_size
        self.input_target_size = input_target_size
        print(self.input_target_size)

    def augment_data(self, image):
        image = tf.image.resize_with_crop_or_pad(image, (self.input_target_size[0] + 30),
                                                 (self.input_target_size[1] + 30))
        # image = tf.image.random_crop(image, size=[self.input_target_size[0],self.input_target_size[1]])
        image = tf.image.random_crop(image, size=[self.input_target_size[0],
                                                  self.input_target_size[1],
                                                  self.input_target_size[2]])
        image = tf.image.random_brightness(image, max_delta=0.5)
        image = tf.image.random_flip_left_right(image, seed=None)

        return image

    def get_classNames(self, ):
        train_dir = pathlib.Path(self.directory)
        CLASSNAME = np.array([item.name for item in train_dir.glob('*') if item.name != "LICENSE.txt"])
        return CLASSNAME

    def get_fullDataset(self):
        print("********* get_fullDataset ***************")

        full_data = tf.data.Dataset.list_files(str(self.directory + "*/*"))
        return full_data

    def getTrainValidationData(self, ):
        print("********* getTrainValidationData ***************")

        dataset_size = len(list(self.get_fullDataset()))

        train_size = int((1 - self.validation_size) * dataset_size)
        train_dataset = self.get_fullDataset().take(train_size)
        validation_dataset = self.get_fullDataset().skip(train_size)

        return train_dataset, validation_dataset

    def get_label(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        return parts[-2] == self.get_classNames()

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
        label = self.get_label(image_path)
        image = self.load_image(image_path)

        return image, label

    def load_image_train(self, image_path):
        image, label = self.load_image_with_label(image_path)
        if self.augmentation:
            image = self.augment_data(image)

        image = self.resize_image(image, self.input_target_size[0],
                                  self.input_target_size[1], )
        image = self.normalize(image)

        return image, label

    def load_image_test(self, image_path):
        image, label = self.load_image_with_label(image_path)
        image = self.resize_image(image, self.input_target_size[0],
                                  self.input_target_size[1], )
        image = self.normalize(image)
        return image, label

    def train_validation_split(self):
        train_dataset, validation_dataset = self.getTrainValidationData()
        train_dataset = train_dataset.map(self.load_image_train)
        train_dataset = train_dataset.shuffle(self.shuffle_buffer_size)
        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.prefetch(1000)

        validation_dataset = validation_dataset.map(self.load_image_test)
        validation_dataset = validation_dataset.cache()

        validation_dataset = validation_dataset.batch(self.batch_size)

        return train_dataset, validation_dataset
