# Chương 6: Công Cụ và Thư Viện Phổ Biến trong Học Máy

## 6.1 Tổng Quan về Công Cụ và Thư Viện trong Học Máy

Học máy là một lĩnh vực rộng và có nhiều công cụ, thư viện phần mềm hỗ trợ giúp cho việc phát triển mô hình dễ dàng và hiệu quả hơn. Các thư viện này cung cấp các công cụ từ tiền xử lý dữ liệu, huấn luyện mô hình, đánh giá mô hình, cho đến triển khai mô hình trong môi trường thực tế.

### Các công cụ và thư viện học máy phổ biến bao gồm:

1. **NumPy**: Thư viện tính toán khoa học với mảng (arrays) mạnh mẽ.
2. **Pandas**: Thư viện xử lý và phân tích dữ liệu với cấu trúc dữ liệu bảng.
3. **Matplotlib** và **Seaborn**: Các thư viện trực quan hóa dữ liệu mạnh mẽ.
4. **Scikit-learn**: Thư viện học máy phổ biến cho các thuật toán học máy cổ điển.
5. **TensorFlow** và **Keras**: Thư viện học sâu (deep learning) với khả năng xây dựng và huấn luyện các mạng nơ-ron sâu.
6. **PyTorch**: Thư viện học sâu mạnh mẽ, được sử dụng phổ biến trong nghiên cứu học sâu.
7. **XGBoost**: Thư viện mạnh mẽ dành cho học máy với thuật toán cây quyết định nâng cao.

---

## 6.2 Các Thư Viện Cơ Bản

### 6.2.1 **NumPy**

**NumPy** là một thư viện nền tảng trong khoa học dữ liệu và học máy, hỗ trợ các mảng đa chiều (ndarrays) và các phép toán trên mảng hiệu quả.

- **Khởi tạo mảng**:

```python
import numpy as np

# Tạo một mảng 1D
a = np.array([1, 2, 3])

# Tạo một mảng 2D
b = np.array([[1, 2], [3, 4]])

print(a)
print(b)
```

- **Các phép toán với mảng**:

```python
# Cộng 2 mảng
c = a + np.array([4, 5, 6])

# Tính tổng tất cả các phần tử
total = np.sum(a)

# Ma trận chuyển vị
transpose_b = np.transpose(b)

print(c)
print(total)
print(transpose_b)
```

### 6.2.2 **Pandas**

**Pandas** là thư viện xử lý và phân tích dữ liệu, hỗ trợ làm việc với dữ liệu dạng bảng (DataFrames).

- **Khởi tạo DataFrame**:

```python
import pandas as pd

# Tạo DataFrame từ dictionary
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Los Angeles', 'Chicago']}

df = pd.DataFrame(data)

print(df)
```

- **Thao tác cơ bản**:

```python
# Lọc dữ liệu
df_filtered = df[df['Age'] > 30]

# Thống kê mô tả
print(df.describe())

# Tính toán trung bình của cột 'Age'
avg_age = df['Age'].mean()

print(df_filtered)
print(avg_age)
```

### 6.2.3 **Matplotlib và Seaborn**

**Matplotlib** và **Seaborn** là hai thư viện phổ biến để trực quan hóa dữ liệu, giúp hiển thị đồ thị như biểu đồ, đồ thị phân tán, histograms, box plots, v.v.

- **Biểu đồ đường và cột**:

```python
import matplotlib.pyplot as plt

# Dữ liệu
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Vẽ biểu đồ đường
plt.plot(x, y)
plt.title('Biểu đồ đường')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

- **Biểu đồ phân tán với Seaborn**:

```python
import seaborn as sns

# Dữ liệu
tips = sns.load_dataset('tips')

# Vẽ biểu đồ phân tán
sns.scatterplot(x='total_bill', y='tip', data=tips)
plt.show()
```

---

## 6.3 Các Thư Viện Học Máy Cổ Điển

### 6.3.1 **Scikit-learn**

**Scikit-learn** là thư viện học máy phổ biến, hỗ trợ nhiều thuật toán học máy cổ điển như hồi quy tuyến tính, cây quyết định, SVM, KNN, v.v. Đây là thư viện phổ biến nhất cho người mới bắt đầu với học máy.

- **Hồi quy tuyến tính**:

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Dữ liệu ví dụ
X = np.random.rand(100, 1)  # Tạo dữ liệu ngẫu nhiên
y = 3 * X + 2 + np.random.randn(100, 1)  # Dữ liệu mục tiêu

# Chia dữ liệu thành bộ huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình
model = LinearRegression()

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
```

- **Cây quyết định**:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# Tải bộ dữ liệu Iris
iris = load_iris()

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Khởi tạo mô hình
model = DecisionTreeClassifier()

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Đánh giá mô hình
accuracy = model.score(X_test, y_test)
print(f'Accuracy: {accuracy}')
```

---

## 6.4 Các Thư Viện Học Sâu (Deep Learning)

### 6.4.1 **TensorFlow và Keras**

**TensorFlow** là một thư viện mạnh mẽ do Google phát triển, chủ yếu được sử dụng cho học sâu và học máy quy mô lớn. **Keras** là một API của TensorFlow giúp xây dựng các mô hình học sâu dễ dàng hơn.

- **Khởi tạo mô hình MLP trong Keras**:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Xây dựng mô hình
model = Sequential()
model.add(Dense(64, input_dim=8, activation='relu'))  # Lớp ẩn
model.add(Dense(1, activation='sigmoid'))  # Lớp đầu ra

# Biên dịch mô hình
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=10, batch_size=32)
```

### 6.4.2 **PyTorch**

**PyTorch** là thư viện học sâu mạnh mẽ, được phát triển bởi Facebook. PyTorch nổi bật nhờ tính linh hoạt và dễ sử dụng, đặc biệt trong nghiên cứu học sâu.

- **Xây dựng mạng nơ-ron đơn giản trong PyTorch**:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Xây dựng mô hình mạng nơ-ron
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Khởi tạo mô hình, loss function và optimizer
model = SimpleNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện mô hình
for epoch in range(10):
    # Giả sử X_train và y_train là dữ liệu đầu vào
    inputs = torch.tensor(X_train, dtype=torch.float32)
    targets = torch.tensor(y_train, dtype=torch.float32)
    
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward pass và tối ưu hóa
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
```

---

## 6.5 Các Thư Viện Phổ

 Biến Khác

### 6.5.1 **XGBoost**

**XGBoost** là thư viện học máy sử dụng thuật toán Gradient Boosting, nổi bật trong các cuộc thi học máy và được ứng dụng rộng rãi trong các bài toán phân loại và hồi quy.

```python
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Tải bộ dữ liệu Boston
data = load_boston()
X = data.data
y = data.target

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình XGBoost
model = xgb.XGBRegressor(objective='reg:squarederror')

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán và đánh giá
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
```

---

## 6.6 Tổng Kết

Các công cụ và thư viện học máy là yếu tố quan trọng giúp bạn xây dựng các mô hình học máy mạnh mẽ. Với các thư viện phổ biến như **NumPy**, **Pandas**, **Matplotlib**, **Scikit-learn**, **TensorFlow**, **PyTorch**, và **XGBoost**, bạn có thể dễ dàng xử lý dữ liệu, huấn luyện các mô hình học máy, và triển khai chúng vào các ứng dụng thực tế. Việc làm quen và thành thạo các công cụ này sẽ giúp bạn phát triển nhanh chóng trong lĩnh vực học máy.