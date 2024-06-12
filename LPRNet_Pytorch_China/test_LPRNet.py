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
    parser.add_argument('--test_img_dirs', default="./data/valid", help='the test images path')
    parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--test_batch_size', default=20, help='testing batch size.')
    parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--show', default=False, type=bool, help='show test image and its predict result or not.')
    parser.add_argument('--pretrained_model', default='./model_trained/LPRNet_Pytorch_China.pth', help='pretrained base model')

    args = parser.parse_args()

    return args

def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.float32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

def test():
    args = get_parser()

    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    print("Successful to build network!")

    # load pretrained model
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model))
        # lprnet.load_state_dict(torch.load(args.pretrained_model, map_location='cpu'))
        print("load pretrained model successful!")
    else:
        print("[Error] Can't found pretrained mode, please check!")
        return False

    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)
    try:
        Greedy_Decode_Eval(lprnet, test_dataset, args)
    finally:
        cv2.destroyAllWindows()

def Greedy_Decode_Eval(Net, datasets, args):
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    total_levenshtein_distance = 0
    t1 = time.time()
    
    sample_results = []

    for i in range(epoch_size):
        images, labels, lengths = next(batch_iterator)
        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start+length]
            targets.append(label)
            start += length
        targets = np.array([el.numpy() for el in targets])
        imgs = images.numpy().copy()

        if args.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        prebs = Net(images)
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
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
            preb_labels.append(no_repeat_blank_label)
        for i, label in enumerate(preb_labels):
            if args.show:
                show(imgs[i], label, targets[i])
            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            if (np.asarray(targets[i]) == np.asarray(label)).all():
                Tp += 1
            else:
                Tn_2 += 1

            # Calculate Levenshtein distance
            pred_str = ''.join([CHARS[idx] for idx in label])
            target_str = ''.join([CHARS[int(idx)] for idx in targets[i]])
            levenshtein_distance_value = levenshtein_distance(pred_str, target_str)
            total_levenshtein_distance += levenshtein_distance_value

            # Collect sample results
            if len(sample_results) < 3:
                sample_results.append((imgs[i], pred_str, target_str, levenshtein_distance_value))

    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    mean_levenshtein_distance = total_levenshtein_distance / (Tp + Tn_1 + Tn_2)
    print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
    print("[Info] Mean Levenshtein Distance: {}".format(mean_levenshtein_distance))
    t2 = time.time()
    print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))

    # Display 3 random samples with predictions and Levenshtein distance
    print("\nSample Predictions and Levenshtein Distances:")
    for img, pred_str, target_str, levenshtein_distance_value in sample_results:
        show(img, [CHARS_DICT[c] for c in pred_str], [CHARS_DICT[c] for c in target_str])
        print("Predicted: {}, Target: {}, Levenshtein Distance: {}".format(pred_str, target_str, levenshtein_distance_value))

# def show(img, label, target):
#     img = np.transpose(img, (1, 2, 0))
#     img *= 128.
#     img += 127.5
#     img = img.astype(np.uint8)

#     lb = "".join([CHARS[idx] for idx in label])
#     tg = "".join([CHARS[int(idx)] for idx in target])

#     flag = "F"
#     if lb == tg:
#         flag = "T"
#     img = cv2ImgAddText(img, lb, (0, 0))
    
#     # Use matplotlib to display the image
#     plt.imshow(img)
#     plt.title(f"Target: {tg} ### {flag} ### Predict: {lb}")
#     plt.axis('off')  # Turn off axis numbers and ticks
#     plt.show()

#     print("target: ", tg, " ### {} ### ".format(flag), "predict: ", lb)
def show(img, label, target, save_dir="output"):
    """
    Lưu ảnh kết quả với nhãn và dự đoán vào folder.

    Args:
        img (np.ndarray): Mảng numpy chứa ảnh (C, H, W).
        label (list): Danh sách các chỉ số ký tự dự đoán.
        target (list): Danh sách các chỉ số ký tự mục tiêu.
        save_dir (str): Đường dẫn đến thư mục lưu ảnh.
    """
    
    img = np.transpose(img, (1, 2, 0))
    img *= 128.
    img += 127.5
    img = img.astype(np.uint8)

    lb = "".join([CHARS[idx] for idx in label])
    tg = "".join([CHARS[int(idx)] for idx in target])

    flag = "F"
    if lb == tg:
        flag = "T"
    
    img = cv2ImgAddText(img, lb, (0, 0))
    
    # Tạo thư mục lưu trữ nếu chưa tồn tại
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Tạo tên file từ target và flag
    filename = f"{tg}_{flag}_{lb}.jpg"
    filepath = os.path.join(save_dir, filename)
    
    # Lưu ảnh vào file
    # plt.imshow(img)
    # plt.title(f"Target: {tg} ### {flag} ### Predict: {lb}")
    # plt.axis('off')
    plt.savefig(filepath)
    plt.close()  # Đóng figure để giải phóng bộ nhớ

    print(f"Đã lưu ảnh vào: {filepath}")
    print("target: ", tg, " ### {} ### ".format(flag), "predict: ", lb)

    
def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=12):
    if (isinstance(img, np.ndarray)):  # detect opencv format or not
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("data/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
    draw.text(pos, text, textColor, font=fontText)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def levenshtein_distance(a, b):
    """Calculates the Levenshtein distance between a and b."""
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n, m)) space
        a, b = b, a
        n, m = m, n

    current_row = range(n + 1)  # Keep current and previous row, not full matrix
    for i in range(1, m + 1):
        previous_row, current_row = current_row, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            current_row[j] = min(add, delete, change)

    return current_row[n]

if __name__ == "__main__":
    test()


