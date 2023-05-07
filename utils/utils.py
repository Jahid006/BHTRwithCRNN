from PIL import Image
import Levenshtein 
import unicodedata


class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            if isinstance(my_dict[key], dict):
                setattr(self, key, Dict2Class(my_dict[key]))
            else:
                setattr(self, key, my_dict[key])


def levenshtein_distance(a,b, normalize=False):
    if normalize:
        a = unicodedata.normalize('NFKC', a)
        b = unicodedata.normalize('NFKC', b)
    return Levenshtein.distance(a,b)


def make_grid(images_list, size=(192, 32), shape=None):
    width, height = size
    images = [
        paste_in_the_middle(Image.open(image[:]), size)
        for image in images_list
    ]
    # Create canvas for the final image with total size
    shape = shape if shape else (1, len(images))
    image_size = (width * shape[1], height * shape[0])
    image = Image.new('RGB', image_size)
    # Paste images into final image
    for row in range(shape[0]):
        for col in range(shape[1]):
            offset = width * col, height * row
            idx = row * shape[1] + col
            image.paste(images[idx], offset)
    return image


def paste_in_the_middle(image: Image, canvas_size: tuple = None):
    canvas = Image.new('L', canvas_size, color='white')
    cw, ch = canvas.size
    image.thumbnail((cw, ch), Image.ANTIALIAS)
    w, h = image.size

    if w < cw:
        canvas.paste(image, ((cw-w)//2, 0))
    elif h < ch:
        canvas.paste(image, (0, ch-h)//2)
    elif w == cw and h == ch:
        canvas = image
    else:
        image.resize((cw, ch))
        canvas = image
    return canvas


def count_parameters(model):
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
