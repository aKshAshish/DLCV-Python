from keras.preprocessing.image import img_to_array

# To preprocess images to return a numpy array with
# row*column*channels format
class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        self.dataFormat = dataFormat

    def preprocess(self, image):
        return img_to_array(image, data_format=self.dataFormat)