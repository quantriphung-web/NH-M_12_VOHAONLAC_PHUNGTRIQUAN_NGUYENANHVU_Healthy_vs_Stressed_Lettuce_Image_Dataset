
#CNN

model = models.Sequential([
    # INPUT: Ảnh RGB 224x224
    layers.Input(shape=(224, 224, 3)),

    #  HỌC ĐẶC TRƯNG CƠ BẢN

    # Tạo 32 kernel tích chập 3x3 để học các đặc trưng cơ bản
    # như cạnh lá, viền lá, sự thay đổi màu sắc
    layers.Conv2D(32, 3, padding='same', activation='relu'),

    # Chuẩn hóa đầu ra của Conv2D giúp mô hình học ổn định hơn
    # Gián tiếp giúp chọn đặc trưng quan trọng
    BatchNormalization(),

    # Conv2D lần 2 giúp tăng khả năng học đặc trưng ở cùng cấp độ
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    BatchNormalization(),

    # Giảm kích thước đặc trưng, giữ lại đặc trưng nổi bật nhất
    # => Feature Selection
    layers.MaxPooling2D(),

    # BLOCK 2 – ĐẶC TRƯNG TRUNG CẤP

    # Tăng số kernel lên 64 để học đặc trưng phức tạp hơn
    # như vân lá, vùng đổi màu
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    BatchNormalization(),

    layers.Conv2D(64, 3, padding='same', activation='relu'),
    BatchNormalization(),

    # Giảm nhiễu, giữ đặc trưng quan trọng
    layers.MaxPooling2D(),

    #ĐẶC TRƯNG CẤP CAO

    # Học đặc trưng phức tạp như:
    # đốm bệnh, vùng héo, tổn thương lá
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    BatchNormalization(),

    layers.Conv2D(128, 3, padding='same', activation='relu'),
    BatchNormalization(),

    layers.MaxPooling2D(),

    # ĐẶC TRƯNG RẤT CAO

    # Học đặc trưng tổng quát toàn bộ lá
    layers.Conv2D(256, 3, padding='same', activation='relu'),
    BatchNormalization(),

    layers.MaxPooling2D(),

    # GOM ĐẶC TRƯNG

    # Chuyển toàn bộ feature map thành vector đặc trưng
    # nhưng giữ thông tin tổng thể
    layers.GlobalAveragePooling2D(),

    # PHÂN LOẠI (KHÔNG PHẢI TẠO ĐẶC TRƯNG)
    # Kết hợp các đặc trưng đã học
    layers.Dense(256, activation='relu'),
    BatchNormalization(),

    # Giảm overfitting
    layers.Dropout(0.5),

    # Dự đoán xác suất cho từng lớp
    layers.Dense(num_classes, activation='softmax')
])




#NAIVE BAYES

# Khởi tạo danh sách lưu dữ liệu và nhãn
X, y = [], []
# Lấy tên các lớp (Healthy, Stressed, ...)
class_names = sorted(os.listdir(DATA_DIR))
for label, class_name in enumerate(class_names):
    class_path = os.path.join(DATA_DIR, class_name)
    # Duyệt từng ảnh trong mỗi lớp
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        # Đọc ảnh bằng OpenCV
        img = cv2.imread(img_path)

        # Nếu ảnh lỗi thì bỏ qua
        if img is None:
            continue

        # Resize ảnh về kích thước cố định 224x224
        img = cv2.resize(img, IMG_SIZE)

        # Chuẩn hóa giá trị pixel về [0,1]
        img = img / 255.0

        # CHUYỂN ẢNH → VECTOR 1 CHIỀU
        # Đây chính là bước tạo đặc trưng
        img = img.flatten()

        # Lưu vector đặc trưng
        X.append(img)

        # Lưu nhãn
        y.append(label)
