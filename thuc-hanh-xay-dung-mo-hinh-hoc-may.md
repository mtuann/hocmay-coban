# Chương 4: Thực Hành: Xây Dựng Mô Hình Học Máy

## 4.1 Tổng Quan Về Quy Trình Xây Dựng Mô Hình Học Máy

Quy trình xây dựng mô hình học máy bao gồm nhiều bước, từ việc thu thập và chuẩn bị dữ liệu cho đến việc huấn luyện, đánh giá và triển khai mô hình. Các bước chính trong quy trình này bao gồm:

1. **Thu thập và chuẩn bị dữ liệu**: Thu thập, làm sạch và chuẩn bị dữ liệu.
2. **Phân tích dữ liệu**: Khám phá và hiểu dữ liệu để có cái nhìn rõ ràng về các đặc trưng và mối quan hệ giữa chúng.
3. **Chia dữ liệu**: Chia dữ liệu thành các bộ huấn luyện và kiểm tra.
4. **Lựa chọn thuật toán và huấn luyện mô hình**: Lựa chọn thuật toán học máy phù hợp và huấn luyện mô hình.
5. **Đánh giá mô hình**: Đánh giá hiệu quả của mô hình bằng các chỉ số như độ chính xác, F1-score, độ chính xác trên bộ kiểm tra.
6. **Triển khai mô hình**: Sau khi mô hình đã hoàn thiện và đánh giá, triển khai mô hình vào thực tế.

---

## 4.2 Chuẩn Bị Dữ Liệu

Trước khi bắt đầu xây dựng mô hình học máy, việc chuẩn bị dữ liệu là bước vô cùng quan trọng. Dữ liệu phải sạch, không có giá trị thiếu (missing values), và các đặc trưng phải được lựa chọn và xử lý đúng cách.

### 4.2.1 Thu Thập Dữ Liệu

Dữ liệu có thể được thu thập từ nhiều nguồn khác nhau, bao gồm:
- **Dữ liệu có sẵn**: Các bộ dữ liệu đã được thu thập và có sẵn trên các nền tảng như Kaggle, UCI Machine Learning Repository.
- **Dữ liệu thu thập từ API**: Dữ liệu có thể thu thập từ các API (ví dụ: Twitter API, Google Maps API).
- **Dữ liệu tự thu thập**: Dữ liệu có thể thu thập từ các thiết bị cảm biến hoặc web scraping.

### 4.2.2 Làm Sạch Dữ Liệu

Quá trình làm sạch dữ liệu bao gồm các bước như:
- **Loại bỏ các giá trị thiếu**: Xử lý các giá trị bị thiếu (missing values) bằng cách thay thế hoặc loại bỏ chúng.
- **Chuyển đổi dữ liệu**: Đảm bảo rằng các giá trị thuộc tính (features) có kiểu dữ liệu đúng (ví dụ: số học, chuỗi, ngày tháng).
- **Chuẩn hóa dữ liệu**: Đảm bảo rằng dữ liệu đầu vào có cùng đơn vị đo lường hoặc chuẩn hóa để cải thiện hiệu quả của thuật toán (ví dụ: chuẩn hóa hoặc chuẩn hóa Z-score).

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Đọc dữ liệu
data = pd.read_csv('data.csv')

# Xử lý giá trị thiếu
data = data.fillna(data.mean())  # Thay thế giá trị thiếu bằng giá trị trung bình của cột

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['feature1', 'feature2', 'feature3']])
```

---

## 4.3 Phân Tích Dữ Liệu

Trước khi huấn luyện mô hình, việc phân tích dữ liệu là rất quan trọng để hiểu các đặc trưng (features) và mối quan hệ giữa chúng.

### 4.3.1 Khám Phá Dữ Liệu (Exploratory Data Analysis - EDA)

EDA là quá trình tìm hiểu về các đặc trưng của dữ liệu, bao gồm:
- **Thống kê mô tả**: Tính toán các chỉ số như trung bình, độ lệch chuẩn, min, max, quartiles.
- **Trực quan hóa dữ liệu**: Sử dụng biểu đồ như biểu đồ histogram, scatter plot, heatmap để hiểu mối quan hệ giữa các đặc trưng.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Hiển thị thống kê mô tả
print(data.describe())

# Vẽ biểu đồ phân phối của các đặc trưng
sns.histplot(data['feature1'], kde=True)
plt.show()

# Vẽ biểu đồ heatmap của ma trận tương quan
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()
```

---

## 4.4 Chia Dữ Liệu

Việc chia dữ liệu thành các bộ huấn luyện và kiểm tra (training and test set) là rất quan trọng để đánh giá chính xác mô hình. Dữ liệu huấn luyện sẽ được sử dụng để huấn luyện mô hình, trong khi dữ liệu kiểm tra sẽ được sử dụng để đánh giá mô hình.

### 4.4.1 Phương Pháp Chia Dữ Liệu

Một trong những phương pháp phổ biến là **chia tỉ lệ 80-20** (80% huấn luyện và 20% kiểm tra). Hoặc có thể sử dụng **K-fold cross-validation** để đánh giá mô hình hiệu quả hơn.

```python
from sklearn.model_selection import train_test_split

# Chia dữ liệu thành bộ huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(data[['feature1', 'feature2']], data['target'], test_size=0.2, random_state=42)
```

---

## 4.5 Lựa Chọn Thuật Toán và Huấn Luyện Mô Hình

Sau khi dữ liệu đã được chuẩn bị, bước tiếp theo là lựa chọn thuật toán học máy phù hợp và huấn luyện mô hình. Dưới đây là các thuật toán phổ biến trong học máy:

### 4.5.1 Hồi Quy Tuyến Tính (Linear Regression)

```python
from sklearn.linear_model import LinearRegression

# Khởi tạo mô hình hồi quy tuyến tính
model = LinearRegression()

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán trên bộ kiểm tra
y_pred = model.predict(X_test)
```

### 4.5.2 Phân Loại Với K-NN

```python
from sklearn.neighbors import KNeighborsClassifier

# Khởi tạo mô hình K-NN
model = KNeighborsClassifier(n_neighbors=3)

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán trên bộ kiểm tra
y_pred = model.predict(X_test)
```

### 4.5.3 Cây Quyết Định

```python
from sklearn.tree import DecisionTreeClassifier

# Khởi tạo mô hình cây quyết định
model = DecisionTreeClassifier()

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán trên bộ kiểm tra
y_pred = model.predict(X_test)
```

---

## 4.6 Đánh Giá Mô Hình

Sau khi huấn luyện mô hình, việc đánh giá mô hình là rất quan trọng để hiểu được hiệu quả của mô hình đối với bài toán.

### 4.6.1 Các Chỉ Số Đánh Giá

Các chỉ số đánh giá mô hình phổ biến gồm:
- **Độ chính xác (Accuracy)**: Tỉ lệ dự đoán đúng.
$$
\text{Accuracy} = \frac{\text{Số lượng dự đoán đúng}}{\text{Tổng số mẫu}}
$$
- **Ma trận nhầm lẫn (Confusion Matrix)**: Hiển thị số lượng dự đoán đúng và sai cho từng lớp.
- **F1-Score**: Trung bình điều hòa của độ chính xác và độ nhạy (recall).
$$
\text{F1-score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$
- **ROC-AUC**: Đo lường khả năng phân biệt giữa các lớp.

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Tính độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Ma trận nhầm lẫn
print(confusion_matrix(y_test, y_pred))

# Báo cáo phân loại
print(classification_report(y_test, y_pred))
```

---

## 4.7 Triển Khai Mô Hình

Sau khi mô hình đã được huấn luyện và đánh giá, bước tiếp theo là triển khai mô hình vào thực tế. Điều này có thể bao gồm:
- **Lưu mô hình**: Lưu mô hình đã huấn luyện để sử dụng lại trong tương lai.
- **Triển khai mô hình lên môi trường sản xuất**: Chạy mô hình trên các máy chủ hoặc dịch vụ web.

```python
import joblib

# Lưu mô hình vào file
joblib.dump(model, 'model.pkl')

# Tải mô hình từ file
model = joblib.load('model.pkl')
```

---

## 4.8 Tóm Tắt

Quy trình xây dựng mô hình học máy gồm nhiều bước quan trọng như chuẩn

 bị dữ liệu, lựa chọn thuật toán, huấn luyện mô hình và đánh giá kết quả. Việc thực hành các bước này sẽ giúp bạn xây dựng các mô hình học máy có hiệu quả và áp dụng chúng vào các bài toán thực tế.

