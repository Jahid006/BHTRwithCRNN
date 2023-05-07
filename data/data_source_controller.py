import random
import json
from collections import namedtuple, defaultdict


class DataSourceController(object):
    def __init__(
        self,
        data: dict = None,
        base_dir: str = '',
        id: str = '',
        take_n: int = None,
        filter: callable = lambda x: True,
        transform: callable = lambda x: x,
        random_state: int = 42
    ):
        self.data = []
        self.format = namedtuple('data', ['id', 'path', 'label'])
        self.ids = defaultdict(int)
        self.filter = filter
        self.transform = transform
        self.random_state = random_state

        if data and base_dir and id:
            self.add_data(data, base_dir, id, take_n)

    def add_data(
        self, data: dict,
        base_dir: str = '', id: str = 'na', take_n=None
    ):

        if isinstance(data, str):
            try:
                data = self.read_json(data)
            except Exception as exp:
                print(exp, 'Data not vaild')

        _data = {
            k: self.transform(v) for k, v in data.items()
            if self.filter(self.format(id, f"{base_dir}/{k}",  data[k]))
        }

        _keys = list(_data.keys())

        if take_n:
            take_n = min(take_n, len(_data))
            _keys = random.sample(_keys, take_n)

        _data = [
            self.format(id, f"{base_dir}/{k}",  _data[k])
            for k in _keys
        ]

        self.data.extend(_data)
        self.ids[id] += len(_data)

        print(f"Out of {len(data)} {id},{len(_data)} are kept after filtering")
        print(f"Total data {len(self.data)}")

    def modifiy_filter(self, function: callable, update_data=False):
        self.filter = function

        if update_data:
            n_data = len(self.data)
            self.data = [d for d in self.data if self.filter(d)]
            self.ids = defaultdict(int)

            for d in self.data:
                self.ids[d.id] += 1

            print(f"Out of {n_data}, {len(self.data)} are kept after filtering")

    def shuffle(self):
        random.shuffle(self.data, random_state=self.random_state)

    def read_json(self, path, encoding='utf-8'):
        return json.load(open(path, 'r', encoding=encoding))

    # def split_data(self, split_names=['train', 'val', 'test'], ratio=[.8, .1, .1]):
    #     self.shuffle()
    #     splits = []
    #     start = 0
    #     for split_name, ration in zip(split_names, ratio):
    #         _temp = DataSourceController(
    #             data=None,
    #             base_url=self.base_dir,
    #             id=split_name,
    #             filter=self.filter,
    #             transform=self.transform,
    #             random_state=self.random_state
    #         )
    #         _temp.data = self.data[start: start+int(ration*len(self.data))]
    #         for idx, t in enumerate(_temp.data):
    #             _temp.data[idx].id = split_name
    #         splits.append(_temp)
    #         start += len(_temp.data)

    #     return splits

    @property
    def unique_source(self):
        return self.ids.keys()

    def __repr__(self) -> str:
        return f"Total Data: {len(self.data)}"+'\n'\
               + '\n'.join([f"{k}: {v}" for k, v in self.ids.items()])