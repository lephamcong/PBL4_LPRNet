## Đề tài: Nhận dạng biển số xe với Deep Neural Networks
## Thành viên :

| **Tên**              | **Lớp**               | **Mã sinh viên**          |
|-------------------    |-------------------------|--------------------------|
| **Lê Phạm Công**    | 20KTMT1                 | 106200221                |
| **Phan Công Danh**       | 20KTMT1                 | 106200222                |
| **Trần Đình Thi**    | 20KTMT1                 | 106200246                |

## Mô tả

Đề tài này nhằm triển khai thực hiện triển khai mô hình LPRNet [1] với bộ dữ liệu biển số xe Trung Quốc được tổng hợp từ bộ dữ liệu CCPD2020 [2] và bộ dữ liệu biển số xe Việt Nam được thu thập thông qua ảnh chụp thực tế.

## Dataset
#### Bộ dữ liệu biển số xe Trung Quốc:
#### Bộ dữ liệu biển số xe Việt Nam:

## Cấu trúc thư mục

- `Dataset/`: Thư mục chứa Bộ dữ liệu biển số xe Trung Quốc và Việt Nam
- `LPRNet_Pytorch_China/`: Chứa source code huấn luyện và đánh giá với Bộ dữ liệu biển số xe Trung Quốc 
- `LPRNet_Pytorch_VietNam/`: Chứa source code huấn luyện và đánh giá với Bộ dữ liệu biển số xe Việt Nam
- `readme.md`: file này
- `requirements.txt`: file thư viện cần thiết (hướng dẫn cài thư viện bên dưới)
## Cách sử dụng

**1. Clone repo:**

```
https://github.com/lephamcong/PBL4_LPRNet.git
```

**2. Cài đặt các thư viện cần thiết:**

```
pip install -r requirements.txt
```
**3. Huấn luyện và đánh giá với bộ dữ liệu biển số xe Trung Quốc**
```
cd LPRNet_Pytorch_China
```
*Huấn luyện (điều chỉnh đường dẫn đến bộ dữ liệu)*
```
python train_LPRNet.py
```
*Đánh giá (Kết quả được hiển thị như file Notebook LPRNet_Pytorch_China.ipynb)*
```
python test_LPRNet.py
```
**4. Huấn luyện và đánh giá với bộ dữ liệu biển số xe Việt Nam**
```
cd LPRNet_Pytorch_VietNam
```
*Huấn luyện (điều chỉnh đường dẫn đến bộ dữ liệu)*
```
python train_LPRNet.py
```
*Đánh giá (Kết quả được hiển thị như file Notebook LPRNet_Pytorch_China.ipynb)*
```
python test_LPRNet.py
```
## Tài liệu tham khảo

[1] Zherzdev, Sergey, and Alexey Gruzdev. "Lprnet: License plate recognition via deep neural networks." arXiv preprint arXiv:1806.10447 (2018).

[2] Xu, Zhenbo, et al.“Towards end-to-end license plate detection and recogni-
tion: A large dataset and baseline." Proceedings of the European conference on
computer vision (ECCV). 2018

[3] https://github.com/lyl8213/Plate_Recognition-LPRnet

[4] https://github.com/mesakarghm/LPRNET

[5] https://github.com/xuexingyu24/License_Plate_Detection_Pytorch.git

## Thông tin liên hệ
- Lê Phạm Công
- Email: lpc051002@gmail.com


