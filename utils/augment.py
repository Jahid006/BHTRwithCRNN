import random
import cv2
import numpy as np
import skimage


# https://theai.codes/computer-vision/a-list-of-the-most-useful-opencv-filters/
# https://gist.github.com/Prasad9/28f6a2df8e8d463c6ddd040f4f6a028a?permalink_comment_id=2933012#gistcomment-2933012
# You could use straug and Albamunations too

def normalizer(function):
    def inner(*args, **kwargs):
        kwargs['image'] = kwargs['image']/255.0
        output = function(*args, **kwargs)
        return output*255.0

    return inner


class Augmentation(object):
    def __init__(self, proablility=0.2):
        self.probability = proablility

    def __call__(self, image: np.ndarray):

        method_list = [
            func for func in dir(self)
            if callable(getattr(self, func))
            and not func.startswith("__")
        ]
        if np.random.rand() < self.probability:
            return getattr(self, random.choice(method_list))(image=image)

        return image

    def otsu_binarization(self, image=None):
        if image.shape == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image_result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return image_result

    def inverse_binarization(self, image=None):
        image = self.otsu_binarization(image=image)
        return 255-image

    @normalizer
    def gaussian(self, image=None):
        return skimage.util.random_noise(image, mode='gaussian', clip=True)

    @normalizer
    def localvar(self, image=None):
        return skimage.util.random_noise(image, mode='localvar')

    @normalizer
    def poisson(self, image=None):
        return skimage.util.random_noise(image, mode='poisson', clip=True)

    @normalizer
    def salt(self, image=None):
        return skimage.util.random_noise(image, mode='salt')

    @normalizer
    def pepper(self, image=None):
        return skimage.util.random_noise(image, mode='pepper')

    @normalizer
    def s_p(self, image=None):
        return skimage.util.random_noise(image, mode='s&p')

    @normalizer
    def speckle(self, image=None):
        return skimage.util.random_noise(image, mode='speckle', clip=True)

    def skip_noise(self, image=None):
        return image

    # from straug.process import AutoContrast
    # def contrast(self, image):
    #     return AutoContrast()(Image.fromarray(image_result))
