import os
import numpy as np
import theano
from scipy import misc
from data_splitter import DataSplitter
from batch_data_provider import BatchDataProvider
import timeit


class DataProvider(object):
    def __init__(self,
                 input_dir_person,
                 input_dir_background,
                 test_percentage_split,
                 validate_percentage_split,
                 batch):
        self.rng = np.random.RandomState(123456)
        self.input_dir_person = input_dir_person
        self.input_dir_background = input_dir_background
        self.person_photos = []
        self.background_photos = []
        self._list_directory()
        self.data_splitter = DataSplitter(self.person_photos,
                                          self.background_photos,
                                          test_percentage_split,
                                          validate_percentage_split)

        training_person_images, training_background_images = self.data_splitter.get_training_set()
        self.batch_data_provider = BatchDataProvider(training_person_images, training_background_images, batch)

    def clear_batch_index(self):
        self.batch_data_provider.clear_batch_index()

    def get_number_of_batches(self):
        return self.batch_data_provider.get_number_of_batches()

    def get_testing_images(self):
        testing_person_images, testing_background_images = self.data_splitter.get_testing_set()
        images_classes_tuple = self._generate_tuple(testing_background_images, testing_person_images)
        images_classes_tuple = self._read_images(images_classes_tuple)
        self.rng.shuffle(images_classes_tuple)
        return zip(*images_classes_tuple)

    def get_validate_images(self):
        validate_person_images, validate_background_images = self.data_splitter.get_validation_set()
        images_classes_tuple = self._generate_tuple(validate_background_images, validate_person_images)
        images_classes_tuple = self._read_images(images_classes_tuple)
        self.rng.shuffle(images_classes_tuple)
        return zip(*images_classes_tuple)

    def get_batch_training_images(self):
        training_person_images, training_background_images = self.batch_data_provider.get_batch_files()
        if training_person_images is not None and training_background_images is not None:
            images_classes_tuple = self._generate_tuple(training_background_images, training_person_images)
            images_classes_tuple = self._read_images(images_classes_tuple)
            self.rng.shuffle(images_classes_tuple)
            return zip(*images_classes_tuple)
        else:
            return None, None

    def _generate_tuple(self, background_images, person_images):
        temp_all_images = person_images[:]
        temp_all_images += background_images
        ones = self._generate_classes(len(person_images), 1)
        zeros = self._generate_classes(len(background_images), 0)
        temp_all_classes = np.hstack((ones, zeros))
        images_classes_tuple = zip(temp_all_images, temp_all_classes)
        return images_classes_tuple

    def _list_directory(self):
        self.person_photos = os.listdir(self.input_dir_person)
        self.background_photos = os.listdir(self.input_dir_background)

    def _read_images(self, images_and_classes):
        temp_image_list = []
        temp_class_list = []
        for image_class_pair in images_and_classes:
            if image_class_pair[1] == 1:
                path = os.path.join(self.input_dir_person, image_class_pair[0])
            elif image_class_pair[1] == 0:
                path = os.path.join(self.input_dir_background, image_class_pair[0])
            else:
                raise Exception('Wrong folder switch')

            temp_image = self._read_and_reshape(path)
            temp_image_list.append(temp_image)
            temp_class_list.append(image_class_pair[1])
        return zip(temp_image_list, temp_class_list)

    @staticmethod
    def _read_and_reshape(path):
        read_image = misc.imread(path)
        if read_image.shape != (256, 256, 3):
            image_rgb = np.zeros((256, 256, 3), dtype='uint8')
            for i in xrange(3):
                image_rgb[:, :, i] = read_image
            read_image = image_rgb

        temp_image = np.asarray(read_image / (256.0, 256.0, 256.0), dtype=theano.config.floatX)
        mean_image = np.mean(temp_image, axis=(0, 1), dtype=theano.config.floatX)
        temp_image -= mean_image
        temp_image = temp_image.transpose(2, 0, 1)
        return temp_image

    @staticmethod
    def _shared_dataset(data_xy, borrow=True):

        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = zip(*data_xy)
        shared_x = theano.shared(np.asarray(data_x, dtype='int8'), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype='int8'), borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, shared_y

    @staticmethod
    def _generate_classes(vector_length, class_number):
        if class_number == 0:
            return np.zeros(vector_length)
        elif class_number == 1:
            return np.ones(vector_length)
        else:
            raise Exception('wrong class number')

if __name__ == '__main__':
    dp = DataProvider(
        input_dir_person='/home/marcin/data/men_detection/men',
        input_dir_background='/home/marcin/data/men_detection/mountains',
        test_percentage_split=10, validate_percentage_split=10, batch=1000)
    start_timer = timeit.default_timer()
    testing_img = dp.get_testing_images()
    val_imgs = dp.get_validate_images()
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
