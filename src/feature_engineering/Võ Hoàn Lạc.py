#Võ Hoàn Lạc
     
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


