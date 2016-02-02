class BatchDataProvider(object):
    def __init__(self, person_images, background_images, batch):
        self.person_images = person_images
        self.background_images = background_images
        self.person_images_len = len(person_images)
        self.background_images_len = len(background_images)
        self.person_first_analysed_element = 0
        self.background_first_analysed_element = 0
        self.batch = batch
        self.class_ratio = self._calculate_class_ratio()
        self.background_index, self.person_index = self._calculate_classes_cardinality()
        self.number_of_batches = self.person_images_len/self.person_index - 1

    def clear_batch_index(self):
        self.person_first_analysed_element = 0
        self.background_first_analysed_element = 0

    def get_number_of_batches(self):
        return self.number_of_batches

    def get_batch_files(self):
        if self._is_batch_out_of_range():
            person_batch_images = self.person_images[self.person_first_analysed_element:
                                                     self.person_first_analysed_element+self.person_index]
            background_batch_images = self.background_images[self.background_first_analysed_element:
                                                             self.background_first_analysed_element +
                                                             self.background_index]
            self.person_first_analysed_element += (self.person_index+1)
            self.background_first_analysed_element += (self.background_index+1)
            return person_batch_images, background_batch_images
        else:
            return None, None

    def _is_batch_out_of_range(self):
        return (self.person_first_analysed_element+self.person_index < self.person_images_len and
                self.background_first_analysed_element+self.background_index < self.background_images_len)

    def _calculate_class_ratio(self):
        return self.person_images_len/float(self.background_images_len)

    def _calculate_classes_cardinality(self):
        background = round(self.batch / (1 + self.class_ratio))
        person = self.batch - background
        return int(background), int(person)
