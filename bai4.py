import os
import cv2  # Thư viện OpenCV để đọc ảnh
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import time

# Thiết lập đường dẫn đến thư mục ảnh
image_folder = 'images'

# Các lớp có sẵn
classes = ['hoa', 'dongvat']  # Thay đổi theo các lớp của bạn
label_map = {cls: idx for idx, cls in enumerate(classes)}  # Tạo nhãn số cho từng lớp

# Đọc và tiền xử lý ảnh
X = []
y = []

for cls in classes:
    cls_folder = os.path.join(image_folder, cls)
    for filename in os.listdir(cls_folder):
        filepath = os.path.join(cls_folder, filename)
        img = cv2.imread(filepath)
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Đổi kích thước ảnh về 128x128
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển ảnh sang ảnh xám (tùy chọn)
            X.append(img.flatten())  # Biến ảnh thành vector
            y.append(label_map[cls])

X = np.array(X)
y = np.array(y)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hàm đánh giá mô hình
def evaluate_model(model, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    print(f"Model: {model.__class__.__name__}")
    print(f"Training Time: {train_time:.4f}s")
    print(f"Prediction Time: {predict_time:.4f}s")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(classification_report(y_test, y_pred))
    print("\n")
    return train_time, predict_time, accuracy, precision, recall

# Các mô hình
models = [
    SVC(kernel='linear'),
    KNeighborsClassifier(n_neighbors=min(3, len(X_train))),
    DecisionTreeClassifier(max_depth=10)
]

# Đánh giá các mô hình
results = {}
for model in models:
    results[model.__class__.__name__] = evaluate_model(model, X_train, X_test, y_train, y_test)

# Tóm tắt kết quả
for model_name, (train_time, predict_time, accuracy, precision, recall) in results.items():
    print(f"{model_name}:")
    print(f"  Training Time: {train_time:.4f}s")
    print(f"  Prediction Time: {predict_time:.4f}s")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print("\n")
