# Học Máy (Machine Learning) - Cơ Bản

Chào mừng bạn đến với kho học liệu về **Machine Learning** (Học Máy). Đây là tài liệu học tập cho những ai mới bắt đầu học về Học Máy, đặc biệt là học sinh cấp 3 và sinh viên năm đầu đại học. Trong kho tài liệu này, chúng ta sẽ tìm hiểu các khái niệm cơ bản, các thuật toán phổ biến, cũng như thực hành với các bài toán thực tế.

## Nội Dung
1. [Giới Thiệu về Học Máy](#giới-thiệu-về-học-máy)
2. [Các Phương Pháp Học Máy](#các-phương-pháp-học-máy)
3. [Các Thuật Toán Cơ Bản](#các-thuật-toán-cơ-bản)
4. [Thực Hành: Xây Dựng Mô Hình Học Máy](#thực-hành-xây-dựng-mô-hình-học-máy)
5. [Công Cụ và Thư Viện Phổ Biến](#công-cụ-và-thư-viện-phổ-biến)
6. [Ứng Dụng Của Học Máy](#ứng-dụng-của-học-máy)
7. [Tài Liệu Tham Khảo](#tài-liệu-tham-khảo)

---

## Giới Thiệu về Học Máy

**Học Máy (Machine Learning)** là một lĩnh vực con trong trí tuệ nhân tạo (AI) giúp máy tính có thể học hỏi và cải thiện khả năng dự đoán hoặc ra quyết định từ dữ liệu mà không cần lập trình chi tiết. Học Máy có thể được chia thành ba loại chính:

1. **Học có giám sát (Supervised Learning)**: Dữ liệu huấn luyện bao gồm các đầu vào và đầu ra, và mục tiêu là học được một hàm ánh xạ từ đầu vào tới đầu ra.
2. **Học không giám sát (Unsupervised Learning)**: Dữ liệu huấn luyện chỉ có đầu vào mà không có đầu ra. Mục tiêu là phát hiện các mẫu hoặc cấu trúc ẩn trong dữ liệu.
3. **Học tăng cường (Reinforcement Learning)**: Máy tính học thông qua tương tác với môi trường và nhận phần thưởng hoặc hình phạt.

---

## Các Phương Pháp Học Máy

- **Học có giám sát**: Trong học có giám sát, mỗi ví dụ trong dữ liệu huấn luyện có một nhãn (label) tương ứng. Mục tiêu là tìm ra mô hình dự đoán chính xác nhãn cho dữ liệu mới. Ví dụ: phân loại (classification), hồi quy (regression).

- **Học không giám sát**: Dữ liệu không có nhãn, và mục tiêu là phát hiện các mẫu hoặc nhóm trong dữ liệu. Ví dụ: phân cụm (clustering), giảm chiều (dimensionality reduction).

- **Học tăng cường**: Trong học tăng cường, một agent học cách tương tác với môi trường để tối đa hóa phần thưởng mà nó nhận được. Ví dụ: chơi game, điều khiển robot.

---

## Các Thuật Toán Cơ Bản

1. **Hồi quy tuyến tính (Linear Regression)**: Đây là một trong những thuật toán cơ bản trong học máy, dùng để dự đoán giá trị liên tục (ví dụ: dự đoán giá nhà).
   
2. **Phân loại Naive Bayes**: Là một thuật toán phân loại dựa trên định lý Bayes, rất hiệu quả với các bài toán phân loại văn bản hoặc email spam.

3. **Cây quyết định (Decision Tree)**: Một thuật toán phân loại và hồi quy có cấu trúc giống cây, nơi mỗi nhánh biểu thị một quyết định hoặc phân loại.

4. **K-Near Neighbors (KNN)**: Là thuật toán phân loại không có tham số, dựa trên việc tìm kiếm k điểm gần nhất trong dữ liệu huấn luyện.

5. **Máy hỗ trợ vector (SVM)**: Là thuật toán phân loại mạnh mẽ, tìm kiếm một hyperplane tối ưu để phân chia các lớp trong không gian dữ liệu.

6. **Mạng nơ-ron nhân tạo (Neural Networks)**: Là mô hình mạnh mẽ có thể học được các quan hệ phức tạp trong dữ liệu. Mạng nơ-ron sâu (Deep Learning) là một dạng nâng cao của mạng nơ-ron.

---

## Thực Hành: Xây Dựng Mô Hình Học Máy

### **1. Cài Đặt Môi Trường**

Trước khi bắt đầu, bạn cần cài đặt một số công cụ và thư viện để thực hành. Dưới đây là cách cài đặt một môi trường Python đơn giản:

```bash
# Cài đặt thư viện cần thiết
pip install numpy pandas scikit-learn matplotlib seaborn
```

### **2. Xây Dựng Mô Hình Hồi Quy Tuyến Tính**

Ví dụ sau đây minh họa cách xây dựng mô hình hồi quy tuyến tính để dự đoán giá nhà.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Tải dữ liệu
data = pd.read_csv('house_prices.csv')

# Chọn các đặc trưng và mục tiêu
X = data[['square_feet', 'num_rooms']]  # Đặc trưng
y = data['price']  # Mục tiêu

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
```

### **3. Phân Loại Với KNN**

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Tải bộ dữ liệu Iris
data = load_iris()
X = data.data
y = data.target

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Xây dựng mô hình KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

---

## Công Cụ và Thư Viện Phổ Biến

- **Python**: Ngôn ngữ lập trình phổ biến trong học máy.
- **Numpy**: Thư viện tính toán số học mạnh mẽ.
- **Pandas**: Thư viện xử lý dữ liệu.
- **Scikit-learn**: Thư viện học máy phổ biến với nhiều thuật toán cơ bản.
- **Matplotlib / Seaborn**: Thư viện vẽ đồ thị và trực quan hóa dữ liệu.
- **TensorFlow / PyTorch**: Các thư viện mạnh mẽ để xây dựng mô hình học sâu (deep learning).

---

## Ứng Dụng Của Học Máy

- **Phân tích dữ liệu và dự báo**: Sử dụng học máy để phân tích dữ liệu và đưa ra dự đoán (ví dụ: dự báo giá trị cổ phiếu).
- **Phân loại hình ảnh**: Chẩn đoán hình ảnh y tế, nhận diện khuôn mặt.
- **Xử lý ngôn ngữ tự nhiên (NLP)**: Dịch máy, phân tích cảm xúc, chatbot.
- **Xe tự lái**: Học máy giúp các phương tiện tự lái nhận diện môi trường và đưa ra quyết định.

---

## Tài Liệu Tham Khảo

- [Cuốn sách "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
- [Khóa học Machine Learning của Andrew Ng trên Coursera](https://www.coursera.org/learn/machine-learning)
- [Documentation của Scikit-learn](https://scikit-learn.org/stable/documentation.html)
- [Machine Learining Cơ Bản - Vũ Hữu Tiệp](https://machinelearningcoban.com/)
---

Chúc bạn học tốt và khám phá thú vị về thế giới của Học Máy! 🚀