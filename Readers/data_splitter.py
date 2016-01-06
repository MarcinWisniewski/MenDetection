import numpy as np


class DataSplitter(object):
    def __init__(self, person_photos, background_photos,
                 test_percentage_split, validate_percentage_split):
        self.rng = np.random.RandomState(123456)
        self.person_photos = person_photos
        self.background_photos = background_photos
        self.test_percentage_split = test_percentage_split
        self.validate_percentage_split = validate_percentage_split

        self._reshuffle_file_names()
        self._calculate_input_length()
        self._calculate_indexes()

    def get_training_set(self):
        return self.person_photos[self.validate_person_images_index:], \
               self.background_photos[self.validate_background_images_index:]

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

    def _calculate_indexes(self):
        self.test_person_images_index = int(self.person_images_length * self.test_percentage_split/100.0)
        self.test_background_images_index = int(self.background_images_length * self.test_percentage_split/100.0)
        self.validate_person_images_index = self.test_person_images_index + \
                                            int(self.person_images_length * self.validate_percentage_split/100.0)
        self.validate_background_images_index = self.test_background_images_index + \
                                                int(self.background_images_length * self.validate_percentage_split/100.0)


if __name__ == '__main__':
    print 1

