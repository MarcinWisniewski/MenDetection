import os
import numpy as np
import theano
from scipy import misc
from data_splitter import DataSplitter
from batch_data_provider import BatchDataProvider
import timeit

_CLASS_FOLDER = ['men', 'mountains']
_DEFAULT_IMAGE_SIZE = (233, 233, 3)


class DataProvider(object):
    def __init__(self,
                 input_dir,
                 test_percentage_split,
                 validate_percentage_split,
                 batch):
        self.rng = np.random.RandomState(123456)
        self.input_dir_person = os.path.join(input_dir, _CLASS_FOLDER[0])
        self.input_dir_background = os.path.join(input_dir, _CLASS_FOLDER[1])
        self.person_photos = []
        self.background_photos = []
        self._list_directory()
        self.data_splitter = DataSplitter(self.person_photos,
                                          self.background_photos,
                                          test_percentage_split,
                                          validate_percentage_split)

        training_person_images, training_background_images = self.data_splitter.get_training_set()
        testing_person_images, testing_background_images = self.data_splitter.get_testing_set()
        validate_person_images, validate_background_images = self.data_splitter.get_validation_set()

        self.training_batch_data_provider = BatchDataProvider(training_person_images, training_background_images, batch)
        self.testing_batch_data_provider = BatchDataProvider(testing_person_images, testing_background_images, batch)
        self.validate_batch_data_provider = BatchDataProvider(validate_person_images, validate_background_images, batch)

    def get_number_of_images(self):
        return len(self.person_photos) + len(self.background_photos)

    def get_number_of_training_batches(self):
        return self.training_batch_data_provider.get_number_of_batches()

    def get_number_of_testing_batches(self):
        return self.testing_batch_data_provider.get_number_of_batches()

    def get_number_of_validate_batches(self):
        return self.validate_batch_data_provider.get_number_of_batches()

    def get_batch_testing_images(self):
        return self._get_image_from_batch_provider(self.testing_batch_data_provider)

    def get_batch_validate_images(self):
        return self._get_image_from_batch_provider(self.validate_batch_data_provider)

    def get_batch_training_images(self):
        return self._get_image_from_batch_provider(self.training_batch_data_provider)

    def _get_image_from_batch_provider(self, batch_data_provider):
        person_images, background_images = batch_data_provider.get_batch_files()
        if person_images is not None and background_images is not None:
            images_classes_tuple = self._generate_image_class_tuple(background_images, person_images)
            images_classes_tuple = self._read_images(images_classes_tuple)
            self.rng.shuffle(images_classes_tuple)
            return zip(*images_classes_tuple)
        else:
            return None, None

    def _generate_image_class_tuple(self, background_images, person_images):
        temp_all_images = person_images[:]
        temp_all_images += background_images
        ones = self._generate_classes(len(person_images), 1)
        zeros = self._generate_classes(len(background_images), 0)
        temp_all_classes = np.hstack((ones, zeros))
        images_classes_tuple = zip(temp_all_images, temp_all_classes)
        return images_classes_tuple

    def _list_directory(self):
        self.person_photos = self._generate_subfolder_image_tuple(self.input_dir_person)
        self.background_photos = self._generate_subfolder_image_tuple(self.input_dir_background)

    def _read_images(self, images_and_classes):
        temp_image_list = []
        temp_class_list = []
        for image_class_pair in images_and_classes:
            if image_class_pair[1] == 1:
                path = os.path.join(self.input_dir_person, image_class_pair[0][0], image_class_pair[0][1])
            elif image_class_pair[1] == 0:
                path = os.path.join(self.input_dir_background, image_class_pair[0][0], image_class_pair[0][1])
            else:
                raise Exception('Wrong folder switch')

            temp_image = self._read_and_reshape(path)
            temp_image_list.append(temp_image)
            temp_class_list.append(image_class_pair[1])
        return zip(temp_image_list, temp_class_list)

    @staticmethod
    def _read_and_reshape(path):
        try:
            read_image = misc.imread(path)
        except IOError:
            print 'broken file', path
            read_image = np.zeros(_DEFAULT_IMAGE_SIZE, dtype='uint8')

        if read_image.shape != _DEFAULT_IMAGE_SIZE:
            image_rgb = np.zeros(_DEFAULT_IMAGE_SIZE, dtype='uint8')
            if read_image.shape == (233, 233):
                for i in xrange(3):
                    image_rgb[:, :, i] = read_image
                read_image = image_rgb
            else:
                read_image = image_rgb
                print path

        read_image = np.asarray(read_image / (256.0, 256.0, 256.0), dtype=theano.config.floatX)
        mean_image = np.mean(read_image, axis=(0, 1), dtype='float')
        read_image -= mean_image
        read_image = read_image.transpose(2, 0, 1)
        return read_image

    @staticmethod
    def _generate_classes(vector_length, class_number):
        if class_number == 0:
            return np.zeros(vector_length, dtype='int32')
        elif class_number == 1:
            return np.ones(vector_length, dtype='int32')
        else:
            raise Exception('wrong class number')

    @staticmethod
    def _generate_subfolder_image_tuple(input_dir):
        subfolders = os.listdir(input_dir)
        all_files = []
        for folder in subfolders:
            image_files = os.listdir(os.path.join(input_dir, folder))
            all_files += [(folder, image) for image in image_files]
        return all_files

if __name__ == '__main__':
    dp = DataProvider(
        input_dir='/home/marcin/data/men_detection',
        test_percentage_split=0.1, validate_percentage_split=0.1, batch=100)
    start_timer = timeit.default_timer()
    testing_img = dp.get_batch_testing_images()
    val_imgs = dp.get_batch_validate_images()
    stop_timer = timeit.default_timer()
    print stop_timer - start_timer

    train_imgs = dp.get_batch_training_images()
    iteration = 0
    while train_imgs != (None, None):
        print iteration
        iteration += 1
        train_start = timeit.default_timer()
        train_imgs = dp.get_batch_training_images()
        train_stop = timeit.default_timer()
        print train_stop - train_start

    print 1
