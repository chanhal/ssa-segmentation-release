import os.path

from .cityscapes import Cityscapes
from .cyclegan import CycleGTA5
from .gta5 import GTA5

from .vaihingen import Vaihingen
from .potsdam import Potsdam


datasets = {
    'cityscapes': Cityscapes,
    'gta5': GTA5,
    'cyclegta5': CycleGTA5,
    'potsdam': Potsdam,
    'vaihingen': Vaihingen,
}


def get_dataset(name, *args, **kwargs):
    return datasets[name](os.path.join('data', name), *args, **kwargs)
