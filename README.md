# **PHẦN 1: HƯỚNG DẪN CÀI ĐẶT & CHẠY TRÊN GOOGLE COLAB**

## **1\. Yêu cầu**

* Tài khoản Google  
* Trình duyệt web  
* Không cần cài Python

## **2\. Mở Google Colab & bật GPU**

1. Truy cập: [https://colab.research.google.com](https://colab.research.google.com)  
2. **Runtime → Change runtime type**  
3. Chọn:  
   * Hardware accelerator: **GPU**  
4. Save

## **3\. Mount Google Drive**

`from google.colab import drive`  
`drive.mount('/content/drive')`

Kiểm tra thư mục:

`import os`  
`print(os.listdir("/content/drive/MyDrive"))`

## **4\. Cấu trúc dữ liệu trong Google Drive**

`MyDrive/`  
`└── datadoan/`  
    `└── plant-health/`  
        `├── healthy/`  
        `└── unhealthy/`

Mỗi class phải là **1 thư mục riêng**

## **5\. Cài đặt thư viện (chạy 1 lần)**

`!pip install -q tensorflow opencv-python scikit-learn seaborn torch torchvision`

## **6\. Chạy chương trình**

1. Copy code vào notebook Colab  
2. Chạy **từ trên xuống dưới**  
3. Thời gian train:  
   * GPU: \~10–20 phút  
   * CPU (không khuyến nghị): rất lâu

## **7\. Dự đoán ảnh mới** 

`from google.colab import files`  
`files.upload()`

#  **PHẦN 2: HƯỚNG DẪN CÀI ĐẶT & CHẠY TRÊN VS CODE (MÁY CÁ NHÂN)**

## **1\. Yêu cầu hệ thống**

### **Tối thiểu:**

* Python **3.9 – 3.11**  
* RAM ≥ 8GB  
* Có GPU là lợi thế (không bắt buộc)

## **2\. Tạo thư mục project**

`plant-health-project/`  
`├── datadoan/`  
`│   └── plant-health/`  
`│       ├── healthy/`  
`│       └── unhealthy/`  
`├── main.py (hoặc notebook.ipynb)`  
`└── requirements.txt`

## **3\. Tạo môi trường ảo (KHUYẾN NGHỊ)**

### **Windows:**

`python -m venv venv`  
`venv\Scripts\activate`

### **macOS / Linux:**

`python3 -m venv venv`  
`source venv/bin/activate`

## **4\. Cài đặt thư viện**

### **Tạo file `requirements.txt`**

`tensorflow`  
`opencv-python`  
`scikit-learn`  
`matplotlib`  
`seaborn`  
`numpy`  
`torch`  
`torchvision`

### **Cài đặt:**

`pip install -r requirements.txt`

 Nếu máy yếu → cài:

`pip install tensorflow-cpu`

## **5\. Chỉnh đường dẫn trong code**

### **Google Colab:**

`DATASET_PATH = "/content/drive/MyDrive/datadoan/plant-health"`

### **VS Code:**

`DATASET_PATH = "./datadoan/plant-health"`

## **6\. Chạy chương trình**

### **Nếu dùng file `.py`**

`python main.py`

### **Nếu dùng Notebook:**

`pip install jupyter`  
`jupyter notebook`

## **7\. Dự đoán ảnh mới (VS Code)**

* Đặt ảnh vào thư mục project  
* Gọi hàm predict trong code  
* Không dùng `files.upload()` (chỉ dùng cho Colab)

