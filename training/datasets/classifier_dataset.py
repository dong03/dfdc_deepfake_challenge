import math
import os
import random
import sys
import traceback
import json
import pdb
import torch
import cv2
import numpy as np
import skimage.draw
from albumentations.augmentations.functional import rot90
from albumentations.pytorch.functional import img_to_tensor
from scipy.ndimage import  binary_dilation
from skimage import measure
from torch.utils.data import Dataset
import dlib

"""
修改：
去掉因diff_mask缺少导致的不能操作的mask
概率：0.35去掉眼睛，0.175去嘴，0.0875去鼻子；0.2去半张脸，0.1是依据diff_mask操作
现为：前四项各0.25

原本只有padding 3（原始尺寸）  修改为random.randint(3,5),三种尺度
"""

def prepare_bit_masks(mask):
    h, w = mask.shape
    mid_w = w // 2
    mid_h = w // 2
    masks = []
    ones = np.ones_like(mask)
    ones[:mid_h] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[mid_h:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:, :mid_w] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:, mid_w:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:mid_h, :mid_w] = 0
    ones[mid_h:, mid_w:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:mid_h, mid_w:] = 0
    ones[mid_h:, :mid_w] = 0
    masks.append(ones)
    return masks


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('libs/shape_predictor_68_face_landmarks.dat')


def blackout_convex_hull(img):
    try:
        out_img = img.copy()
        rect = detector(out_img)[0]
        sp = predictor(out_img, rect)
        landmarks = np.array([[p.x, p.y] for p in sp.parts()])
        outline = landmarks[[*range(17), *range(26, 16, -1)]]
        Y, X = skimage.draw.polygon(outline[:, 1], outline[:, 0])
        cropped_img = np.zeros(out_img.shape[:2], dtype=np.uint8)
        cropped_img[Y, X] = 1
        # if random.random() > 0.5:
        #     img[cropped_img == 0] = 0
        #     #leave only face
        #     return img

        y, x = measure.centroid(cropped_img)
        y = int(y)
        x = int(x)

        first = random.random() > 0.5
        if random.random() > 0.5:
            if first:
                cropped_img[:y, :] = 0
            else:
                cropped_img[y:, :] = 0
        else:
            if first:
                cropped_img[:, :x] = 0
            else:
                cropped_img[:, x:] = 0

        out_img[cropped_img > 0] = 0
        return out_img
    except Exception as e:
        return img


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def remove_eyes(image, landmarks):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[:2]
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    line = binary_dilation(line, iterations=dilation)
    image[line, :] = 0
    return image


def remove_nose(image, landmarks):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[:2]
    x3, y3 = landmarks[2]
    mask = np.zeros_like(image[..., 0])
    x4 = int((x1 + x2) / 2)
    y4 = int((y1 + y2) / 2)
    line = cv2.line(mask, (x3, y3), (x4, y4), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    line = binary_dilation(line, iterations=dilation)
    image[line, :] = 0
    return image


def remove_mouth(image, landmarks):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[-2:]
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 3)
    line = binary_dilation(line, iterations=dilation)
    image[line, :] = 0
    return image


def remove_landmark(image, landmarks):
    if random.random() > 0.67:
        image = remove_eyes(image, landmarks)
    elif random.random() > 0.33:
        image = remove_mouth(image, landmarks)
    else:
        image = remove_nose(image, landmarks)
    return image


def change_padding(image, part=5):
    h, w = image.shape[:2]
    # original padding was done with 1/3 from each side, too much
    pad_h = int(((3 / 5) * h) / part)
    pad_w = int(((3 / 5) * w) / part)
    image = image[h // 5 - pad_h:-h // 5 + pad_h, w // 5 - pad_w:-w // 5 + pad_w]
    return image


class DeepFakeClassifierDataset(Dataset):

    def __init__(self,
                 annotations,
                 label_smoothing=0.01,
                 hardcore=True,
                 normalize={"mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225]},
                 rotation=False,
                 mode="train",
                 balance=True,
                 transforms=None
                 ):
        super().__init__()
        self.mode = mode
        self.rotation = rotation
        self.padding_part = random.randint(3,5)
        self.hardcore = hardcore
        self.label_smoothing = label_smoothing
        self.normalize = normalize
        self.transforms = transforms
        self.balance = balance
        if self.balance:
            self.data = [[x for x in annotations if x[1] == lab] for lab in [0,1]]
            print("neg: %d |pos: %d"%(len(self.data[0]),len(self.data[1])))
        else:
            self.data = [annotations]
            print("all: %d"%len(self.data[0]))
    def load_sample(self,img_path):
        try:
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.mode == "train" and self.hardcore and not self.rotation:
                landmark_path = img_path.split('.')[0] + '.json'
                # 0.7的概率随机去除landmarks，done
                if os.path.exists(landmark_path) and random.random() < 0.75:
                    landmarks = self.parse_json(landmark_path)
                    image = remove_landmark(image, landmarks)
                # 0.2的概率去除整张脸
                elif random.random() < 0.25:
                    image = blackout_convex_hull(image)

            # 裁剪掉人脸周围的空白
            if self.mode == "train" and self.padding_part > 3:
                image = change_padding(image, self.padding_part)
            rotation = 0
            if self.transforms:
                data = self.transforms(image=image)
                image = data["image"]

            if self.mode == "train" and self.rotation:
                rotation = random.randint(0, 3)
                image = rot90(image, rotation)

            image = img_to_tensor(image, self.normalize)
            return image, rotation
        except:
            pdb.set_trace()

    def __getitem__(self, index: int):
        if self.balance:

            safe_idx = index % len(self.data[0])
            img_path_neg = self.data[0][safe_idx][0]
            img_neg,rotation_neg = self.load_sample(img_path_neg)
            lab_neg = self.data[0][safe_idx][1]
            if self.mode == "train":
                lab_neg = np.clip(lab_neg, self.label_smoothing, 1 - self.label_smoothing)

            safe_idx = index % len(self.data[1])
            img_path_pos = self.data[1][safe_idx][0]
            img_pos,rotation_pos = self.load_sample(img_path_pos)
            lab_pos = self.data[1][safe_idx][1]
            if self.mode == "train":
                lab_pos = np.clip(lab_pos, self.label_smoothing, 1 - self.label_smoothing)
            return torch.tensor([lab_neg, lab_pos]), torch.cat((img_neg.unsqueeze(0), img_pos.unsqueeze(0))), \
                   [img_path_neg, img_path_pos]

        else:
            lab = self.data[0][index][1]
            img_path = self.data[0][index][0]
            img, rotation = self.load_sample(img_path)
            lab = torch.tensor(lab,dtype=torch.long)
            return lab, img, img_path#, rotation

    def __len__(self) -> int:
        return max([len(subset) for subset in self.data])

    def parse_json(self,json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        y1, x1, y2, x2 = data['coordinates']
        landmarks = [(data['landmarks'][i] - x1, data['landmarks'][i + 1] - y1) for i in range(0, 10, 2)]
        return landmarks

    def reset_seed(self,epoch,seed):
        seed = (epoch + 1) * seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # cpu
        torch.cuda.manual_seed_all(seed)  # gpu
        torch.backends.cudnn.deterministic = True

def collate_function(data):
    transposed_data = list(zip(*data))
    lab, img, img_path = transposed_data[0], transposed_data[1], transposed_data[2]
    img = torch.stack(img, 0)
    lab = torch.stack(lab, 0)
    return lab, img, img_path