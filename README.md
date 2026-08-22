# CÁCH CHẠY CODE

Toàn bộ chạy trên terminal

## 1. Cài đặt python và môi trường

### 1.1. Cài đặt và cập nhật sang phiên bản Python 3.10 (Nếu môi trường hiện tại chạy bằng phiên bản khác)

**Bước 1**: Chạy đoạn mã sau để cập nhật danh sách gói và cài đặt Python 3.10:

```bash
sudo apt-get update -y
sudo apt-get install python3.10 -y
```

**Bước 2**: Sử dụng `update-alternatives` để thiết lập Python 3.10 làm ưu tiên cao nhất cho cả lệnh `python` và `python3`.

```bash
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
sudo update-alternatives --set python3 /usr/bin/python3.10

sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
sudo update-alternatives --set python /usr/bin/python3.10
```

**Bước 3**: Khi đổi phiên bản Python, trình quản lý gói `pip` cũng cần được cấu hình lại để tương thích với bản 3.10.

```bash
sudo apt-get install python3.10-distutils -y
wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py
```

**Bước 4**: Xác nhận lại xem hệ thống đã nhận diện đúng Python 3.10 chưa.

```bash
python --version
pip --version
```

**Bước 5**: Sau khi hệ thống đã nhận diện đúng Python 3.10 và xác nhận cài đặt được `pip`, thực hiện tải môi trường về.

```bash
pip install -r requirements.txt
```

## 2. Huấn luyện và đánh giá

### 2.1. Huấn luyện

Trong file `train.py` có viết các đối số, cùng với dạng và giá trị mặc định của chúng như sau:

```python
parser.add_argument("--train_data", type=str, required=True)
parser.add_argument("--val_data", type=str, required=True)
parser.add_argument("--image_dir", type=str, required=True)
parser.add_argument("--val_image_dir", type=str, required=True)
parser.add_argument("--checkpoint_dir", type=str, default="./models")
parser.add_argument("--img_size", type=int, default=416)
parser.add_argument("--grid_size", type=int, default=13)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--workers", type=int, default=4)
```

Trong đó:

- `train_data`: đường dẫn đến file dữ liệu huấn luyện.
- `val_data`: đường dẫn đến file dữ liệu kiểm định.
- `image_dir`: đường dẫn đến folder ảnh huấn luyện.
- `val_image_dir`: đường dẫn đến folder ảnh kiểm định. 
- `checkpoint_dir`: đường dẫn đến folder để lưu trọng số. 
- `img_size`: kích thước ảnh.
- `grid_size`: kích thước ma trận lưới.
- `batch_size`: kích thước mỗi batch dữ liệu. 
- `epochs`: số epoch huấn luyện.
- `lr`: viết tắt của learning rate (tốc độ học).
- `workers`: Số luồng/tiến trình của CPU được sử dụng song song để đọc ảnh từ ổ cứng và thực hiện biến đổi dữ liệu trước khi huấn luyện.

Lệnh huấn luyện mô hình theo các đối số mặc định:

```bash
python train.py \
  --train_data ./indoor5-v2-student/public/annotations/train.json \
  --val_data ./indoor5-v2-student/public/annotations/val.json \
  --image_dir ./indoor5-v2-student/public/train/images \
  --val_image_dir ./indoor5-v2-student/public/val/images \
  --checkpoint_dir ./models/
```

Nếu muốn thay đổi các đối số khác với mặc định, ta chỉ cần điền thêm đối số đó và giá trị mà mình mong muốn.

*VD: Nếu muốn thay đổi số epoch thành 20:*

```bash
python train.py \
  --train_data ./indoor5-v2-student/public/annotations/train.json \
  --val_data ./indoor5-v2-student/public/annotations/val.json \
  --image_dir ./indoor5-v2-student/public/train/images \
  --val_image_dir ./indoor5-v2-student/public/val/images \
  --checkpoint_dir ./models/ \
  --epochs 20
```

### 2.2. Suy luận và đánh giá

Trong file `predict.py` có viết các đối số, cùng với dạng và giá trị mặc định của chúng như sau:

```python
parser.add_argument("--image_dir", type=str, required=True, help="Thư mục chứa ảnh cần dự đoán")
parser.add_argument("--output", type=str, default="predictions.json")
parser.add_argument("--checkpoint_dir", type=str, default="./models")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Đường dẫn cụ thể tới file .pth (ưu tiên hơn checkpoint_dir)")
parser.add_argument("--conf_thres", type=float, default=0.3)
parser.add_argument("--iou_thres", type=float, default=0.45)
```

Trong đó:

- `image_dir`: đường dẫn đến folder ảnh kiểm định. 
- `output`: tên file được dùng để ghi lại kết quả dự đoán của mô hình.
- `checkpoint_dir`: đường dẫn đến folder đã lưu trọng số. 
- `checkpoint`: đường dẫn cụ thể đến file trọng số.
- `conf_thres`: ngưỡng độ tin cậy của dự đoán lớp. 
- `iou_thres`: ngưỡng IoU.

Lệnh suy luận theo các đối số mặc định: 

```bash
python predict.py \
  --image_dir ./indoor5-v2-student/public/val/images \
  --output val_predictions.json
```

Nếu muốn thay đổi các đối số khác với mặc định, ta chỉ cần điền thêm đối số đó và giá trị mà mình mong muốn.

*VD: Nếu muốn thay đổi ngưỡng độ tin cậy thành 0.01:*

```bash
python predict.py \
  --image_dir ./indoor5-v2-student/public/val/images \
  --output val_predictions.json \
  --conf_thres 0.01
```

Sau khi thu được file kết quả suy luận `val_predictions.json`, ta thực hiện đánh giá thông qua lệnh sau:

```bash
python ./indoor5-v2-student/public/tools/evaluate_predictions.py \
  --ground_truth ./indoor5-v2-student/public/annotations/val.json \
  --predictions val_predictions.json \
  --output val_score.json
```

Trong đó: 

- `ground_truth`: đường dẫn tới file dữ liệu kiểm định.
- `predictions`: đường dẫn tới file ghi lại kết quả dự đoán. 
- `output`: tên file ghi lại điểm của mô hình.

Ngoài ra, nếu muốn theo dõi kết quả phát hiện ở từng ảnh, ta có thể chạy file `inference.py`, khi đó các ảnh kiểm định sẽ được gán thêm những hộp phát hiện kèm lớp của nó và được lưu về folder `visualized_preds`.

```bash
python inference.py
```