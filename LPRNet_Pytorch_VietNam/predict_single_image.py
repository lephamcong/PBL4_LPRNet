from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from PIL import Image, ImageDraw, ImageFont
from model.LPRNet import build_lprnet
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import time
import cv2
import os

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--test_img_path', default="./data/valid/京PL3N67_0.jpg", help='the test image path') 
    parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
    parser.add_argument('--cuda', default=False, type=bool, help='Use cuda to train model') # Thay đổi default cuda=False
    parser.add_argument('--pretrained_model', default='./model_trained/LPRNet_Pytorch_China.pth', help='pretrained base model')

    args = parser.parse_args()

    return args

def predict_single_image(Net, img_path, args):
    # Load image
    img = cv2.imread(img_path)
    img = cv2.resize(img, tuple(args.img_size))
    img = np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32)
    img -= 127.5
    img *= 0.0078125
    img = np.array(img).reshape((1, 3, args.img_size[1], args.img_size[0]))

    # Convert to tensor
    image = torch.from_numpy(img)

    if args.cuda:
        image = Variable(image.cuda())
    else:
        image = Variable(image)

    # Predict
    preb = Net(image)
    preb = preb.cpu().detach().numpy()

    # Decode prediction
    preb = preb[0, :, :]
    preb_label = list()
    for j in range(preb.shape[1]):
        preb_label.append(np.argmax(preb[:, j], axis=0))

    no_repeat_blank_label = list()
    pre_c = preb_label[0]
    if pre_c != len(CHARS) - 1:
        no_repeat_blank_label.append(pre_c)
    for c in preb_label:
        if (pre_c == c) or (c == len(CHARS) - 1):
            if c == len(CHARS) - 1:
                pre_c = c
            continue
        no_repeat_blank_label.append(c)
        pre_c = c

    # Get predicted string
    predicted_str = ''.join([CHARS[idx] for idx in no_repeat_blank_label])

    # Lấy nhãn từ tên file ảnh
    label = os.path.basename(img_path).split('_')[0]

    print(f"Label: {label}, Predict: {predicted_str}")


if __name__ == "__main__":
    args = get_parser()

    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    print("Successful to build network!")

    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model, map_location=device)) # Load model với map_location
        print("load pretrained model successful!")
    else:
        print("[Error] Can't found pretrained mode, please check!")

    # Predict on single image
    predict_single_image(lprnet, args.test_img_path, args)