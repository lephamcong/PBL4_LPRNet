from torch.utils.data import Dataset
from imutils import paths
import numpy as np
import random
import cv2
import os

# Biển số xe Việt Nam
CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-']

CHARS_DICT = {char: i for i, char in enumerate(CHARS)}

class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len, PreprocFun=None):
        self.img_dir = img_dir
        self.img_paths = [el for dir in img_dir for el in paths.list_images(dir)]
        random.shuffle(self.img_paths)
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        self.PreprocFun = PreprocFun if PreprocFun is not None else self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        image = cv2.imread(filename)
        image = cv2.resize(image, self.img_size) if (image.shape[0], image.shape[1]) != self.img_size else image
        image = self.PreprocFun(image)

        # Tách tên file và lấy nhãn
        basename = os.path.basename(filename).split('_')[0].split('.')[0]
        label = [CHARS_DICT[char] for char in basename if char in CHARS_DICT]

        if not 7 <= len(label) <= 9:
            print(f"Invalid label length for {basename}")
            return None  # Xử lý tình huống nhãn không hợp lệ

        return image, label, len(label)

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        return img

    def check(self, label):
        # Hàm này có thể bị loại bỏ hoặc thay đổi theo nhu cầu kiểm tra mới
        return True


