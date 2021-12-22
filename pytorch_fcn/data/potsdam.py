import os.path

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from collections import namedtuple

# import pdb


Label = namedtuple( 'Label' , ['name', 'trainId', 'color'])
potsdam_labels = [
    Label("Impervious surfaces", 255, np.array((255, 255, 255)).reshape((1,1,3))),
    Label("Building",            0,   np.array((0, 0, 255)).reshape((1,1,3))),
    Label("Low vegetation",      1,   np.array((0, 255, 255)).reshape((1,1,3))),
    Label("Tree",                2,   np.array((0, 255, 0)).reshape((1,1,3))),
    Label("Car",                 3,   np.array((255, 255, 0)).reshape((1,1,3))),
    Label("Clutter/background",  4,   np.array((255, 0, 0)).reshape((1,1,3))),
]


def remap_labels_to_train_ids(arr):
    """
    arr: rgb image
    """
    masks = []
    for label in potsdam_labels:
        mask = np.sum((arr == label.color), axis=2)        
        masks.append(mask == 3)
    
    train_id_map = np.zeros(arr.shape[:2], np.uint8)
    for label, mask in zip(potsdam_labels, masks):
        train_id_map[mask] = label.trainId

    return train_id_map


class Potsdam(data.Dataset):

    num_classes = 6

    def __init__(self, root, split='train', remap_labels=True, transform=None,
                 target_transform=None):
        self.root = root
        self.split = split
        self.remap_labels = remap_labels
        self.ids = self.collect_ids()
        self.transform = transform
        self.target_transform = target_transform

    def collect_ids(self):
        im_dir = os.path.join(self.root, 'rgb', self.split)
        ids = []
        for dirpath, dirnames, filenames in os.walk(im_dir):
            for filename in filenames:
                if filename.endswith('.tif'):
                    ids.append(filename)
        return ids

    def img_path(self, id):
        fmt = 'rgb/{}/{}'
        path = fmt.format(self.split, id)
        return os.path.join(self.root, path)

    def label_path(self, id):
        id = id.replace("RGB", "label")

        fmt = 'gt/{}/{}'
        path = fmt.format(self.split, id)
        return os.path.join(self.root, path)

    def __getitem__(self, index):
        
        # pdb.set_trace()

        id = self.ids[index]
        img = Image.open(self.img_path(id)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = Image.open(self.label_path(id))
        if self.remap_labels:
            target = np.asarray(target)
            target = remap_labels_to_train_ids(target)
            target = Image.fromarray(target, 'L')

            # target.save("d:\\0tmp\\" + id)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.ids)



if __name__=="__main__":

    # pdb.set_trace()
    dataset = Potsdam(r"D:\d_data\00Z\Potsdam")
    dataset.__getitem__(11)