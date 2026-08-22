import json
import cv2
import os


def visualize_predictions(image_dir, pred_file, output_dir, conf_threshold=0.2):
    """
    Đọc file JSON dự đoán và vẽ Bounding Box lên ảnh thực tế.
    """
    # 1. Tạo thư mục chứa ảnh kết quả nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load dữ liệu từ file JSON
    print(f"Đang đọc dữ liệu từ {pred_file}...")
    with open(pred_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    count = 0
    # 3. Duyệt qua từng ảnh trong file dự đoán
    for item in predictions:
        img_name = item.get("image_id")
        img_path = os.path.join(image_dir, img_name)

        if not os.path.exists(img_path):
            continue

        # Đọc ảnh bằng OpenCV
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        boxes = item.get("boxes", [])
        has_boxes = False

        # 4. Vẽ từng Bounding Box lên ảnh
        for box_info in boxes:
            conf = box_info.get("confidence", 0)

            # Bỏ qua các dự đoán có độ tin cậy quá thấp
            if conf < conf_threshold:
                continue

            has_boxes = True
            bbox = box_info.get("bbox")  # Định dạng: [xmin, ymin, xmax, ymax]
            cls_name = str(box_info.get("class", "Unknown"))

            # Ép kiểu toạ độ về số nguyên để OpenCV có thể vẽ
            xmin, ymin, xmax, ymax = map(int, bbox)

            # Vẽ viền khung chữ nhật (Màu Xanh lá - BGR: 0, 255, 0)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Tạo nền và viết chữ (Nhãn + Độ tin cậy)
            label = f"{cls_name} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(img, (xmin, ymin - 20), (xmin + w, ymin), (0, 255, 0), -1)
            cv2.putText(img, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 5. Chỉ lưu những ảnh có ít nhất 1 dự đoán hợp lệ để bạn đỡ phải tìm kiếm
        if has_boxes:
            out_path = os.path.join(output_dir, img_name)
            cv2.imwrite(str(out_path), img)
            count += 1

    print(f"Hoàn tất! Đã vẽ và lưu {count} ảnh vào thư mục: {output_dir}")


if __name__ == '__main__':
    # Chạy hàm với các đường dẫn của bạn
    visualize_predictions(
        image_dir="./indoor5-v2-student/public/val/images",
        pred_file="val_predictions.json",
        output_dir="visualized_preds",
        conf_threshold=0.1  # Set thấp (0.1) để xem mô hình đang "nghĩ" gì ở mức độ thấp nhất
    )