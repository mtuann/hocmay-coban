# Chương 3: Các Thuật Toán Cơ Bản trong Học Máy

## 3.1 Tổng Quan Về Các Thuật Toán Cơ Bản

Trong học máy, có rất nhiều thuật toán khác nhau để giải quyết các bài toán phân loại, hồi quy, phân cụm và giảm chiều dữ liệu. Các thuật toán cơ bản được sử dụng rộng rãi trong học máy có thể chia thành các nhóm chính như sau:

1. **Thuật toán hồi quy (Regression Algorithms)**.
2. **Thuật toán phân loại (Classification Algorithms)**.
3. **Thuật toán phân cụm (Clustering Algorithms)**.
4. **Thuật toán giảm chiều dữ liệu (Dimensionality Reduction Algorithms)**.

Trong chương này, chúng ta sẽ đi vào chi tiết một số thuật toán cơ bản của từng nhóm trên.

---

## 3.2 Thuật Toán Hồi Quy (Regression Algorithms)

Hồi quy là phương pháp dự đoán một giá trị liên tục dựa trên một hoặc nhiều biến đầu vào. Đây là một trong những bài toán cơ bản trong học máy.

### 3.2.1 Hồi Quy Tuyến Tính (Linear Regression)

**Định Nghĩa**: Hồi quy tuyến tính là một thuật toán học máy dùng để mô hình hóa mối quan hệ giữa một biến phụ thuộc $y$ và một hoặc nhiều biến độc lập $x_1, x_2, ..., x_n$.

**Công thức mô hình hồi quy tuyến tính**:
$$
y = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b
$$
Trong đó:
- $y$ là giá trị dự đoán,
- $x_1, x_2, ..., x_n$ là các đặc trưng đầu vào (features),
- $w_1, w_2, ..., w_n$ là các trọng số (weights),
- $b$ là hệ số chệch (bias).

**Mục tiêu**: Tìm các trọng số $w$ sao cho tổng sai số giữa giá trị thực tế và giá trị dự đoán là nhỏ nhất. Sai số này thường được tính bằng **sai số bình phương trung bình** (Mean Squared Error, MSE):
$$
MSE = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2
$$
Trong đó $m$ là số mẫu và $\hat{y}_i$ là giá trị dự đoán của mẫu $i$.

---

## 3.3 Thuật Toán Phân Loại (Classification Algorithms)

Phân loại là một dạng bài toán trong học máy trong đó mục tiêu là phân nhóm dữ liệu vào một trong nhiều lớp (class).

### 3.3.1 K-Nearest Neighbors (K-NN)

**Định Nghĩa**: K-Nearest Neighbors (K-NN) là một thuật toán phân loại (hoặc hồi quy) trong đó mỗi điểm dữ liệu được phân loại theo các điểm lân cận của nó. Số lượng điểm lân cận được chỉ định bởi tham số $k$.

**Công thức tính khoảng cách Euclidean** giữa hai điểm $x_i$ và $x_j$:
$$
d(x_i, x_j) = \sqrt{\sum_{k=1}^{n} (x_{ik} - x_{jk})^2}
$$
Trong đó $x_i = (x_{i1}, x_{i2}, ..., x_{in})$ và $x_j = (x_{j1}, x_{j2}, ..., x_{jn})$ là hai điểm trong không gian đặc trưng.

**Quy trình**:
1. Tính khoảng cách giữa điểm cần phân loại với tất cả các điểm trong tập huấn luyện.
2. Chọn $k$ điểm gần nhất.
3. Phân loại điểm dựa trên nhãn của các điểm này (chọn nhãn xuất hiện nhiều nhất).

---

### 3.3.2 Support Vector Machine (SVM)

**Định Nghĩa**: SVM là một thuật toán phân loại mạnh mẽ tìm kiếm siêu phẳng tối ưu phân chia các lớp trong không gian đặc trưng.

**Công thức SVM**:
SVM tìm kiếm siêu phẳng $w^T x + b = 0$ sao cho khoảng cách giữa các điểm của hai lớp là lớn nhất. Hàm mục tiêu của SVM là tối thiểu hóa:
$$
\text{minimize} \, \frac{1}{2} \| w \|^2
$$
Điều này đảm bảo rằng khoảng cách giữa siêu phẳng và các điểm gần nhất (các điểm biên) là lớn nhất.

**Mục tiêu**:
- Phân chia các lớp sao cho khoảng cách giữa các điểm của hai lớp là lớn nhất.

---

### 3.3.3 Quyết Định Cây (Decision Trees)

**Định Nghĩa**: Cây quyết định là một thuật toán phân loại (hoặc hồi quy) trong đó mỗi nút trong cây đại diện cho một quyết định dựa trên một đặc trưng của dữ liệu.

**Cấu trúc cây quyết định**:
- Mỗi nhánh phân tách dữ liệu theo một điều kiện (ví dụ: $x_1 \leq 5$).
- Các lá (leaf nodes) chứa nhãn lớp hoặc giá trị dự đoán.

**Công thức đánh giá chất lượng phân tách**: Cây quyết định sử dụng các chỉ số như **entropy** hoặc **Gini index** để đánh giá sự phân tách dữ liệu:
- **Entropy**: Được sử dụng trong **ID3** và **C4.5**.
$$
Entropy(S) = - \sum_{i=1}^{n} p_i \log_2(p_i)
$$
Trong đó $p_i$ là xác suất của lớp $i$ trong tập dữ liệu $S$.

- **Gini Index**: Được sử dụng trong **CART (Classification and Regression Trees)**.
$$
Gini(S) = 1 - \sum_{i=1}^{n} p_i^2
$$
Trong đó $p_i$ là xác suất của lớp $i$.

---

## 3.4 Thuật Toán Phân Cụm (Clustering Algorithms)

Phân cụm là quá trình nhóm các đối tượng có đặc điểm tương tự vào cùng một nhóm (cluster). Dưới đây là một số thuật toán phân cụm phổ biến.

### 3.4.1 K-Means Clustering

**Định Nghĩa**: K-Means là thuật toán phân cụm phổ biến nhất. Thuật toán này phân chia dữ liệu thành $k$ cụm sao cho tổng khoảng cách giữa các điểm trong cùng cụm là nhỏ nhất.

**Quy trình K-Means**:
1. Chọn $k$ centroid (trung tâm cụm) ban đầu.
2. Gán mỗi điểm dữ liệu vào cụm gần nhất.
3. Cập nhật các centroid bằng trung bình của các điểm trong cụm.
4. Lặp lại bước 2 và 3 cho đến khi không có thay đổi nào.

**Công thức tính khoảng cách Euclidean** (tương tự như trong K-NN):
$$
d(x_i, c_k) = \| x_i - c_k \|^2
$$
Trong đó $c_k$ là centroid của cụm $k$.

---

### 3.4.2 DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

**Định Nghĩa**: DBSCAN là thuật toán phân cụm dựa trên mật độ, không yêu cầu xác định số lượng cụm $k$ trước. Nó có thể phát hiện các cụm có hình dạng bất kỳ và có khả năng xử lý nhiễu (outliers).

**Các tham số**:
- **Epsilon ($\epsilon$)**: Khoảng cách tối đa giữa hai điểm trong cùng một cụm.
- **MinPts**: Số lượng điểm tối thiểu trong một vùng có mật độ cao để tạo thành một cụm.

**Quy trình**:
1. Chọn một điểm dữ liệu ngẫu nhiên.
2. Tìm tất cả các điểm trong bán kính $\epsilon$ từ điểm đã chọn.
3. Nếu số điểm trong bán kính $\epsilon$ lớn hơn hoặc bằng MinPts, nhóm chúng lại thành một cụm.
4. Tiếp tục với các điểm chưa được phân nhóm.

---

## 3.5 Thuật Toán Giảm Chiều Dữ Liệu (Dimensionality Reduction Algorithms)

### 3.5.1 Principal Component Analysis (PCA)

**Định Nghĩa**: PCA là một kỹ thuật giảm chiều dữ liệu mạnh mẽ, giúp biến đổi các đặc trưng ban đầu thành các thành phần chính (principal components) mà vẫn giữ lại được hầu hết thông tin.

**Quy trình PCA**:
1. Chuẩn hóa dữ liệu.
2. Tính ma trận hiệp phương sai.
3. Tính các vector riêng (eigenvectors) và giá trị riêng (eigenvalues) của ma trận hiệp phương sai.
4. Chọn các vector riêng có giá trị riêng lớn nhất.
5.

 Dự đoán các thành phần chính của dữ liệu.

---

## 3.6 Tóm Tắt

Trong học máy, các thuật toán cơ bản như hồi quy tuyến tính, K-NN, SVM, cây quyết định, K-Means và PCA là những công cụ quan trọng giúp chúng ta giải quyết các bài toán từ phân loại, hồi quy, phân cụm cho đến giảm chiều dữ liệu. Mỗi thuật toán có những đặc điểm riêng và phù hợp với những bài toán khác nhau. Việc hiểu rõ về các thuật toán này sẽ giúp bạn xây dựng được các mô hình học máy hiệu quả trong thực tế.