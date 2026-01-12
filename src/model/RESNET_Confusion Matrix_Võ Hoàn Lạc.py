#Võ Hoàn Lạc
     

#Tiền xử lý dữ liệu
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
#  Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

DATA_DIR = "/content/drive/MyDrive/datadoan/plant-health" # Corrected path to the class subfolders

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = dataset.classes
print("Classes:", class_names)

     
Mounted at /content/drive
Classes: ['healthy', 'unhealthy']

# Load ảnh và gán label
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define IMG_SIZE as it's used in this cell but not yet defined in the kernel state.
IMG_SIZE = (224, 224)

data = []
labels = []
#Resize về 244x244
#Gán : Healthy = 0, Stressed = 1
for label, cls in enumerate(class_names):
  #Xác định đường dẫn thư mục ảnh
    folder = os.path.join(DATA_DIR, cls)
    #Kiểm tra thư mục có tồn tại, nếu có bỏ qua tiếp tục
    if not os.path.isdir(folder):
        print(f"Warning: Directory not found: {folder}")
        continue
        #Duyệt ảnh trong từng thư mục và lấy danh sách tên các file ảnh trong thư mục
    for img_name in os.listdir(folder):
        #Tạo đường dẫn đầy đủ tới đến từng ảnh
        img_path = os.path.join(folder, img_name)
         #Đọc ảnh bằng OpenCV
        img = cv2.imread(img_path)
       #Kiểm tra anh đọc có thành công không
       #Nếu ảnh sai định dạng bỏ qua, in báo cáo
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        #Chuyển không gian màu BGR thành RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #Risize về kích thước chuẩn, đưa tất cả về kích thước 244x244
        img = cv2.resize(img, IMG_SIZE) # Fixed: Changed (IMG_SIZE, IMG_SIZE) to IMG_SIZE
        #Lưu ảnh đã xử lý vào date, lưu nhãn vào label
        data.append(img)
        labels.append(label)

X = np.array(data, dtype="float32")
y = np.array(labels)

     

X = X / 255.0
#Sau khi đọc ảnh, dữ liệu được chuẩn hoá bằng cách chia giá trị pixel cho 255.0 nhằm đưa tất cả giá trị về khoảng [0, 1].
#Việc chuẩn hoá giúp quá trình huấn luyện mô hình học sâu ổn định hơn, tăng tốc độ hội tụ và phù hợp với các mô hình pretrained như ResNet.

     

#CHIA TRAIN / VALIDATION / TEST
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

     

#TĂNG CƯỜNG DỮ LIỆU
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    #Xoay ảnh ngẫu nhiên trong khoảng -30 đến 30 độ
    rotation_range=30,
    #Phóng to thu anh nhỏ ngẫu nhiêu từ -20% đến 20%
    zoom_range=0.2,
    #Lật ảnh theo chiều ngang
    horizontal_flip=True
)

datagen.fit(X_train)

     

#TRÍCH XUẤT ĐẶC TRƯNG BẰNG RESNET50
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
#KHởi tạo ResNet50
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)
#Không cập nhập trọng số ResNet trong quán trình train
for layer in base_model.layers:
    layer.trainable = False


#Gắn classifier
#Lấy đầu ra
x = base_model.output
x = Flatten()(x)
#Chống Overfitting
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
#Phân loại nhị phân
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

     

#COMPILE & TRAIN
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=20
)

     
#FINE-TUNING RESNET
import tensorflow as tf
#Chỉ mở 30 layer cuối của ResNet
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=10
)
     


#ĐÁNH GIÁ MÔ HÌNH
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

     

#Confusion Matrix & Report
from sklearn.metrics import confusion_matrix, classification_report

y_pred = (model.predict(X_test) > 0.5).astype(int)

print(classification_report(y_test, y_pred, target_names=class_names))
print(confusion_matrix(y_test, y_pred))

     
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title("Accuracy")
  plt.legend(["Train","Validation"])
  plt.show()

     
