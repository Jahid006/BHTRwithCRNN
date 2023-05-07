import os
from dataclasses import dataclass
import torch
from PIL import Image
import numpy as np

from BnTokenizer.base import BnGraphemizer
from trainer.sequence_decoder import ctc_decode
from modeling.crnn import CRNN


@dataclass
class Config:
    id: str = ""
    image_width: int = 32
    image_height: int = 128
    logging_path: os.path = None
    pretrainied_model_path: os.path = "artifacts/crnn/CRNN+GRAPHEMIZER+BTHR+Boise/crnn_044500_loss_0.8612_acc_0.8996.pt"
    batch_size: int = 128
    workers: int = 8
    device: str = 'cuda'
    decoding_method: int = 'beam_search'
    beam_size: int = 10
    pickled_tokenizer_path: os.path = './tokenizers/tokenizer_object_2145_valid.pkl'


@dataclass
class Datum:
    id: str = 'N\A'
    path: os.path = ''
    label: str = ''


def get_model(
    config,
    num_class
):
    crnn = CRNN(
        1,
        num_class,
        map_to_seq_hidden=config['map_to_seq_hidden'],
        rnn_hidden=config['rnn_hidden'],
        leaky_relu=config['leaky_relu'],
        # cnn_config=config
    )
    return crnn


def get_tokenizer(
    pickled_tokenizer_path: os.path, 
    char_tokenizer: bool = False
):
    

    tokenizer = BnGraphemizer(max_len=28, normalize_unicode=True)
    tokenizer.load(pickled_tokenizer_path)

    if char_tokenizer:
        char_vocab = set(''.join(tokenizer.vocab))
        tokenizer = BnGraphemizer(max_len=30, normalize_unicode=True)
        tokenizer.add_tokens(char_vocab, reset_oov=True)

    return tokenizer


class InferenceAPI:
    def __init__(
        self,
        config: Config,
        tokenizer: BnGraphemizer,
        model: torch.nn.Module
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.model = model

    def _paste_in_the_middle(self, image: Image, canvas: Image):
        cw, ch = canvas.size
        image.thumbnail((cw, ch), Image.ANTIALIAS)
        w, h = image.size

        if w < cw:
            canvas.paste(image, ((cw-w) // 2, 0))
        elif h < ch:
            canvas.paste(image, (0, (ch-h) // 2))
        elif w == cw and h == ch:
            canvas = image
        else:
            image.resize((cw, ch))
            canvas = image
        return canvas

    def preprocess(self, image: Image):
        image = image.convert('L')
        canvas = Image.new(
            'L',
            (self.config.image_height, self.config.image_width),
            color='white'
        )

        image = self._paste_in_the_middle(
            image=image,
            canvas=canvas
        )

        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        image = (image / 127.5) - 1.0

        return image

    def forward(self, images: torch.FloatTensor):
        with torch.no_grad():
            logits = self.model(images.to(self.config.device))
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        preds = ctc_decode(
            log_probs.detach(),
            method=self.config.decoding_method,
            beam_size=self.config.beam_size
        )

        return self.tokenizer.ids_to_text(preds)

    def predict(self, image: Image):
        image = self.preprocess(image)
        image = torch.FloatTensor(np.array([image]))
        text = self.forward(images=image)

        return text


if __name__ == "__main__":

    config = Config()
    tokenizer = get_tokenizer(config.pickled_tokenizer_path)
    model = get_model(config, len(tokenizer.vocab)+1)

    model.load_state_dict(
        torch.load(config.pretrainied_model_path, map_location=config.device)['model']
    )
    model.to(config.device)
    model.eval()
    print("Model Loaded Successfully")

    inference_api = InferenceAPI(
        config=config,
        tokenizer=tokenizer,
        model=model
    )

    print(inference_api.predict(
        image=Image.open(
            "/home/jahid/Downloads/1.jpg"
            )
        )
    )
