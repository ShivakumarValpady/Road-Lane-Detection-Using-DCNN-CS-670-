from torch.utils.data import Dataset
import os
import json
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import setup

class dataset(Dataset):
    def __init__(self, root_dir:str, sub_dirs:list, data_labels:list, n_frames:int=5, shuffle=True, shuffle_seed=7):
        self.root_dir = root_dir
        self.sub_dirs = sub_dirs
        assert n_frames>=1 and n_frames<20
        self.n_frames = n_frames
        labels = [json.loads(line) for d_label in data_labels for line in open(d_label, 'r')]
        self.labels = {self.get_clip_id(label['raw_file']): label for label in labels}
        self.sample_folders = []
        for sub_folder in sub_dirs:
            for sample_folder in os.listdir(os.path.join(root_dir, sub_folder)):
                if not sample_folder.startswith('.'):
                    self.sample_folders.append(os.path.join(sub_folder, sample_folder))

        assert len(self.sample_folders) == len(self.labels)
        if shuffle:
            random.seed(shuffle_seed)
            random.shuffle(self.sample_folders)

    def __len__(self):
        return len(self.sample_folders)

    def __getitem__(self, idx):

        images_dir = os.path.join(self.root_dir, self.sample_folders[idx])
        frames = []
        for img_name in range(1, 21)[-self.n_frames:]:
            img_name = str(img_name) + ".jpg"
            frames.append(cv2.imread(os.path.join(images_dir, img_name)))

        label_key = self.sample_folders[idx]
        label = self.labels[label_key]

        assert label_key == label['raw_file'].replace('clips/', '').replace('/20.jpg', '')
        lanes = label['lanes']
        height = label['h_samples']
        target_ = self.get_scaled_target(lanes, height, frames[0].shape) 
        target_ = self.to_tensor(target_)
        frames = [self.to_tensor(self.resize(f)/255.) for f in frames] 

        return frames, target_

    def get_scaled_target(self, lanes:list, heights:list, original_shape:tuple, new_dim=(128, 256, 1)):

        gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, heights) if x >= 0] for lane in lanes]
        image = np.zeros(new_dim).astype(np.float32)

        c = (1., 1., 1.)   # white lane points (normalized)

        scale_factorY = new_dim[0] * (1./original_shape[0])
        scale_factorX = new_dim[1] * (1./original_shape[1])
        for lane in gt_lanes_vis:
            lane = [(x*scale_factorX, y*scale_factorY) for (x, y) in lane]
            cv2.polylines(image, np.int32([lane]), isClosed=False, color=c, thickness=3)
        imagee = image.astype(np.float32)

        return imagee

    def get_clip_id(self, filename:str):
        if filename.endswith('.jpg'):
            chunks = filename.split('/')
            return '/'.join([chunks[1], chunks[2]])
        else:
            pass

    def resize(self, image, new_dim=(256, 128)):
        return cv2.resize(image, new_dim).astype(np.float32)

    def to_tensor(self, image):
        return torch.from_numpy(image.transpose((2, 0, 1)))


def show_plain_images(images, n_frames, save=False, fname=False):

    plt.figure(num=None, figsize=(20, 4), dpi=80)
    for i in range(n_frames):
        ax = plt.subplot(1, n_frames, i + 1)
        if images[i].shape[0] > 1:
            # print(images[i].size())
            image = images[i].permute(1, 2, 0).numpy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image)
        else:
            ax.imshow(images[i].squeeze().numpy(), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    if save and fname:
        plt.savefig(os.path.join(setup.output_dir, fname), dpi=200)
    else:
        plt.show()
