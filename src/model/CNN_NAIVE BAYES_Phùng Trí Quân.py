#Phùng Trí Quân
     

# IMPORT THƯ VIỆN CHUNG
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization

# MOUNT GOOGLE DRIVE
from google.colab import drive
drive.mount('/content/drive')

# CẤU HÌNH CHUNG
DATA_DIR = "/content/drive/MyDrive/datadoan/plant-health"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 50
VAL_SPLIT = 0.2

#  CNN THUẦN

# 1.TIỀN XỬ LÝ 
train_datagen = ImageDataGenerator(
    rescale=1./255,                 # Chuẩn hóa pixel ảnh từ [0,255] → [0,1]
    rotation_range=10,              # Xoay ảnh ngẫu nhiên trong khoảng ±10 độ
    width_shift_range=0.05,         # Dịch ảnh theo chiều ngang tối đa 5%
    height_shift_range=0.05,        # Dịch ảnh theo chiều dọc tối đa 5%
    zoom_range=0.05,                # Phóng to / thu nhỏ ảnh tối đa 5%
    horizontal_flip=True,           # Lật ngang ảnh ngẫu nhiên
    validation_split=VAL_SPLIT      # Chia dữ liệu thành train và validation
)

# Generator cho validation 
val_datagen = ImageDataGenerator(
    rescale=1./255,                 # Chỉ chuẩn hóa ảnh
    validation_split=VAL_SPLIT      # Dùng cùng tỉ lệ chia dữ liệu
)

# 2. TẠO DỮ LIỆU TRAIN

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,                       # Thư mục gốc chứa dữ liệu
    target_size=IMG_SIZE,           # Resize ảnh về kích thước IMG_SIZE (vd: 224x224)
    batch_size=BATCH_SIZE,          # Số ảnh trong mỗi batch
    class_mode='categorical',       # Nhãn one-hot (phù hợp softmax)
    subset='training',              # Lấy tập train
    shuffle=True                    # Xáo trộn dữ liệu khi huấn luyện
)


# 3. TẠO DỮ LIỆU VALIDATION

val_gen = val_datagen.flow_from_directory(
    DATA_DIR,                       # Thư mục dữ liệu
    target_size=IMG_SIZE,           # Resize ảnh
    batch_size=BATCH_SIZE,          # Batch size
    class_mode='categorical',       # One-hot label
    subset='validation',            # Lấy tập validation
    shuffle=False                   # Không shuffle để đánh giá chính xác
)

# Lấy số lượng lớp từ dữ liệu train
num_classes = train_gen.num_classes
print("Số lớp:", num_classes)

# 4. XÂY DỰNG MÔ HÌNH CNN
model = models.Sequential([

    # Layer input: ảnh RGB có kích thước IMG_SIZE
    layers.Input(shape=(*IMG_SIZE, 3)),

    #  CONV 1 
    layers.Conv2D(32, 3, activation='relu', padding='same'),  # 32 kernel 3x3
    BatchNormalization(),                                     # Chuẩn hóa batch
    layers.MaxPooling2D(),                                    # Giảm kích thước feature map

    # CONV 2 
    layers.Conv2D(64, 3, activation='relu', padding='same'),  # 64 kernel
    BatchNormalization(),                                     # Ổn định gradient
    layers.MaxPooling2D(),                                    # Giảm chiều dữ liệu

    # CONV 3 
    layers.Conv2D(128, 3, activation='relu', padding='same'), # 128 kernel
    BatchNormalization(),                                     # Chuẩn hóa
    layers.MaxPooling2D(),                                    # Downsampling

    # CHUYỂN FEATURE MAP → VECTOR 
    layers.GlobalAveragePooling2D(),                           # Giảm overfitting hơn Flatten

    # FULLY CONNECTED 
    layers.Dense(256, activation='relu'),                      # Dense layer 256 neuron
    layers.Dropout(0.5),                                       # Dropout 50% để tránh overfitting

    # OUTPUT 
    layers.Dense(num_classes, activation='softmax')            # Softmax cho phân loại đa lớp
])

# Hiển thị kiến trúc mô hình
model.summary()

# 5. COMPILE MÔ HÌNH
model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-4),   # Adam optimizer, learning rate = 0.0003
    loss='categorical_crossentropy',             # Loss cho bài toán phân loại đa lớp
    metrics=['accuracy']                          # Đánh giá bằng accuracy
)

# 6. CALLBACKS
early_stop = EarlyStopping(
    monitor='val_loss',           # Theo dõi val_loss
    patience=10,                  # Dừng nếu không cải thiện sau 10 epoch
    restore_best_weights=True     # Trả lại trọng số tốt nhất
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',           # Theo dõi val_loss
    factor=0.3,                   # Giảm learning rate còn 30%
    patience=4,                   # Sau 4 epoch không cải thiện
    min_lr=1e-6                   # Learning rate nhỏ nhất
)

# 7. HUẤN LUYỆN MÔ HÌNH
history = model.fit(
    train_gen,                                # Dữ liệu train
    validation_data=val_gen,                 # Dữ liệu validation
    epochs=EPOCHS,                           # Số epoch
    callbacks=[early_stop, reduce_lr],       # Callback tối ưu
    verbose=1                                # Hiển thị quá trình train
)



# NAIVE BAYES CLASSIFIER
# Chia dữ liệu train / validation
from sklearn.model_selection import train_test_split

# Import mô hình Naive Bayes Gaussian
from sklearn.naive_bayes import GaussianNB

# Các hàm đánh giá mô hình
from sklearn.metrics import accuracy_score, classification_report

# Thư viện xử lý ảnh
import cv2

# 1. ĐỌC DỮ LIỆU ẢNH & TRÍCH XUẤT ĐẶC TRƯNG
X, y = [], []                         # X: đặc trưng ảnh, y: nhãn

# Lấy danh sách tên các lớp (tên thư mục con)
class_names = sorted(os.listdir(DATA_DIR))

# Duyệt từng lớp
for label, class_name in enumerate(class_names):
    class_path = os.path.join(DATA_DIR, class_name)  # Đường dẫn tới thư mục lớp

    # Bỏ qua nếu không phải thư mục
    if not os.path.isdir(class_path):
        continue

    # Duyệt từng ảnh trong thư mục lớp
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)  # Đường dẫn ảnh

        img = cv2.imread(img_path)                     # Đọc ảnh bằng OpenCV
        if img is None:                                # Nếu ảnh lỗi thì bỏ qua
            continue

        img = cv2.resize(img, IMG_SIZE)                # Resize ảnh về IMG_SIZE
        img = img / 255.0                              # Chuẩn hóa pixel về [0,1]
        img = img.flatten()                            # Chuyển ảnh 2D/3D → vector 1D

        X.append(img)                                  # Lưu vector đặc trưng
        y.append(label)                                # Lưu nhãn tương ứng

# Chuyển list sang numpy array
X = np.array(X)
y = np.array(y)


# 2. CHIA DỮ LIỆU TRAIN / VALIDATION
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=VAL_SPLIT,        # Tỉ lệ validation
    random_state=42,            # Seed để tái lập kết quả
    stratify=y                  # Giữ tỉ lệ các lớp không đổi
)


# 3. HUẤN LUYỆN MÔ HÌNH NAIVE BAYES
nb_model = GaussianNB()         # Khởi tạo mô hình Gaussian Naive Bayes
nb_model.fit(X_train, y_train) # Huấn luyện mô hình