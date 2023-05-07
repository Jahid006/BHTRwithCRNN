import random
import numpy as np
from PIL import Image
import torch


class TextDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        img_height,
        img_width,
        max_token_len=30,
        skip_oov_word=True,
        noiseAugment=None,
        training=True
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.img_height = img_height
        self.img_width = img_width
        self.max_token_len = max_token_len
        self.skip_oov_word = skip_oov_word
        self.noiseAugment = noiseAugment
        self.canvas = Image.new('L', (img_width, img_height), color='white')
        self.training = training

        print(f"Total {len(data)} Images found!!!")

    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, index):
        # namedtuple('data', ['id','path','label']
        example = self.data[index]
        text = example.label
        id = example.id
        path = example.path

        try:
            image = Image.open(example.path).convert('L')
        except Exception as e:
           #print(e, example.path)
            self._logging(index)
            return self[index + 1]

        tokens = self.tokenizer.tokenize(text, padding=False)['input_ids']
        if (
            (self.tokenizer.word2index[self.tokenizer.oov_token] in tokens and self.skip_oov_word)
            or len(tokens) > self.max_token_len
        ):
            self._logging(index)
            return self[index + 1]
        
        if self.noiseAugment and self.training:
            image = np.array(image)
            image = self.noiseAugment(image)
            image = Image.fromarray(image.astype(np.uint8))
            # image.save(f"./temp/{id}_{text}_{path.split('/')[-1]}")

        image = self.paste_image(image.copy(), self.canvas.copy())
        image = np.array(image)

        image = np.expand_dims(image, axis=0)
        image = (image / 127.5) - 1.0
        image = torch.FloatTensor(image)

        target = tokens
        target_length = [len(target)]

        target = torch.LongTensor(target)
        target_length = torch.LongTensor(target_length)

        return image, target, target_length, f"{id}|{text}|{path}"

    def paste_image(self, image: Image, canvas: Image):
        cw, ch = canvas.size
        image.thumbnail((cw, ch), Image.ANTIALIAS)
        w, h = image.size

        if w < cw:
            canvas.paste(image, (int((cw-w)*random.random()), 0)) 
        elif h < ch:
            canvas.paste(image, (0, int((ch-h)*random.random())))
        elif w == cw and h == ch:
            canvas = image
        else:
            image.resize((cw, ch))
            canvas = image
        return canvas

    def _logging(self, index):
        file_object = open('log.txt', 'a')
        n = "\n"
        d = self.data[index]
        file_object.write(f"{d.id}\t{d.path}\t{d.label}{n}")
        file_object.close()
