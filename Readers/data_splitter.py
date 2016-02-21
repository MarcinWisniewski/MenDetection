import numpy as np

REDUCED_TRAINING_EXAMPLES = 50000


class DataSplitter(object):
    def __init__(self, person_photos, background_photos,
                 test_percentage_split, validate_percentage_split, reduce_training_set=False):

        self.rng = np.random.RandomState(123456)
        self.person_photos = person_photos
        self.background_photos = background_photos
        self.test_percentage_split = test_percentage_split
        self.validate_percentage_split = validate_percentage_split

        self._reshuffle_file_names()
        self._calculate_input_length()
        self._calculate_indexes(reduce_training_set)

    def get_training_set(self):
        return self.person_photos[self.validate_person_images_index:self.train_person_images_index], \
               self.background_photos[self.validate_background_images_index:self.train_background_images_index]

    def get_testing_set(self):
        return self.person_photos[0:self.test_person_images_index], \
               self.background_photos[0:self.test_background_images_index]

    def get_validation_set(self):
        return self.person_photos[self.test_person_images_index:self.validate_person_images_index], \
               self.background_photos[self.test_background_images_index:self.validate_background_images_index]

    def _reshuffle_file_names(self):
        self.rng.shuffle(self.person_photos)
        self.rng.shuffle(self.background_photos)

    def _calculate_input_length(self):
        self.person_images_length = len(self.person_photos)
        self.background_images_length = len(self.background_photos)

    def _calculate_indexes(self, reduce_training_set):
        if self.test_percentage_split < 1 and self.validate_percentage_split < 1:
            self.test_person_images_index = int(self.person_images_length * self.test_percentage_split)
            self.test_background_images_index = int(self.background_images_length * self.test_percentage_split)
            self.validate_person_images_index = self.test_person_images_index + \
                                                int(self.person_images_length * self.validate_percentage_split)
            self.validate_background_images_index = self.test_background_images_index + \
                                                int(self.background_images_length *
                                                    self.validate_percentage_split)
            if reduce_training_set:
                self.train_background_images_index = self.validate_background_images_index + REDUCED_TRAINING_EXAMPLES
                self.train_person_images_index = self.validate_person_images_index + REDUCED_TRAINING_EXAMPLES
            else:
                self.train_background_images_index = -1
                self.train_person_images_index = -1
        else:
            class_ratio = self.person_images_length/float(self.background_images_length)
            self.test_background_images_index = round(self.test_percentage_split / (1 + class_ratio))
            self.test_person_images_index = self.test_percentage_split-self.test_background_images_index
            self.validate_background_images_index = self.test_background_images_index + \
                                                    round(self.validate_percentage_split / (1 + class_ratio))
            self.validate_person_images_index = self.test_person_images_index + \
                                                self.validate_percentage_split-self.validate_background_images_index



