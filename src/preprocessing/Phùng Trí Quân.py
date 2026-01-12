# ===============================
# TIỀN XỬ LÝ DỮ LIỆU
# ===============================

import os              # Thư viện làm việc với thư mục, đường dẫn file
import cv2             # OpenCV dùng để đọc và xử lý ảnh
import numpy as np     # NumPy dùng để xử lý mảng số liệu

X, y = [], []          # X: lưu dữ liệu ảnh, y: lưu nhãn (label) của ảnh

# Lấy danh sách các lớp (tên thư mục), sắp xếp để nhãn cố định
class_names = sorted(os.listdir(DATA_DIR))

# Duyệt qua từng lớp (mỗi lớp tương ứng một thư mục ảnh)
for label, class_name in enumerate(class_names):

    # Tạo đường dẫn đầy đủ tới thư mục của lớp hiện tại
    class_path = os.path.join(DATA_DIR, class_name)

    # Nếu không phải là thư mục thì bỏ qua
    if not os.path.isdir(class_path):
        continue

    # Duyệt qua từng ảnh trong thư mục lớp
    for img_name in os.listdir(class_path):

        # Ghép đường dẫn đầy đủ tới file ảnh
        img_path = os.path.join(class_path, img_name)

        #  Đọc ảnh từ file bằng OpenCV
        img = cv2.imread(img_path)

        # Nếu ảnh bị lỗi hoặc không đọc được thì bỏ qua
        if img is None:
            continue

        #  Resize ảnh về kích thước cố định (IMG_SIZE)
        # Mục đích: đảm bảo tất cả ảnh có cùng kích thước
        img = cv2.resize(img, IMG_SIZE)

        #  Chuẩn hóa giá trị pixel
        # Chuyển giá trị pixel từ [0–255] về [0–1]
        # Giúp mô hình học ổn định hơn
        img = img / 255.0

        # Flatten ảnh
        # Chuyển ảnh từ dạng ma trận 2D/3D sang vector 1 chiều
        # Bắt buộc vì Gaussian Naive Bayes chỉ làm việc với vector đặc trưng
        img = img.flatten()

        #  Lưu ảnh đã xử lý vào danh sách X
        X.append(img)

        #  Lưu nhãn tương ứng vào danh sách y
        y.append(label)

#  Chuyển danh sách X, y sang NumPy array
# Định dạng chuẩn để dùng với thư viện sklearn
X = np.array(X)
y = np.array(y)
