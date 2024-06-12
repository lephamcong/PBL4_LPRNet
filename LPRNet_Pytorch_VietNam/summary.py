from model.LPRNet import build_lprnet
from torchsummary import summary
import argparse
from data.load_data import CHARS

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', default=150, help='epoch to train the network')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--train_img_dirs', default="./data/biensoxevn/train/train", help='the train images path')
    parser.add_argument('--test_img_dirs', default="./data/biensoxevn/valid/valid", help='the test images path')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--learning_rate', default=0.001, help='base value of learning rate.')
    parser.add_argument('--lpr_max_len', default=9, help='license plate number max length.')
    parser.add_argument('--train_batch_size', default=32, help='training batch size.')
    parser.add_argument('--test_batch_size', default=20, help='testing batch size.')
    parser.add_argument('--phase_train', default=True, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=False, type=bool, help='Use cuda to train model')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--save_interval', default=2000, type=int, help='interval for save model state dict')
    parser.add_argument('--test_interval', default=2000, type=int, help='interval for evaluate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=2e-5, type=float, help='Weight decay for SGD')
    parser.add_argument('--lr_schedule', default=[10, 20, 40, 60, 80, 100], help='schedule for learning rate.') 
    parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
    # parser.add_argument('--pretrained_model', default='./weights/Final_LPRNet_model.pth', help='pretrained base model')
    parser.add_argument('--pretrained_model', default='', help='pretrained base model')

    args = parser.parse_args()

    return args

args = get_parser()

lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)

# Chuyển mô hình lên GPU nếu args.cuda là True
if args.cuda:
    lprnet = lprnet.cuda()

summary(lprnet, input_size=(3, 24, 94), device="cuda" if args.cuda else "cpu") 
