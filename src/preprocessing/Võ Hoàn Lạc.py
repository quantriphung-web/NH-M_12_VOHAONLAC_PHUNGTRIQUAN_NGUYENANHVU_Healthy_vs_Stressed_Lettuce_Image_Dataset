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

     
