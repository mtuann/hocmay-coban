# Chương 2: Các Phương Pháp Học Máy

## 2.1 Tổng Quan Về Các Phương Pháp Học Máy

Học máy có thể được phân loại theo nhiều cách khác nhau. Trong chương này, chúng ta sẽ tập trung vào ba nhóm phương pháp học máy cơ bản: **Học có giám sát**, **Học không giám sát** và **Học tăng cường**. Mỗi nhóm có các kỹ thuật và phương pháp riêng biệt, phù hợp với các bài toán khác nhau trong thực tế.

## 2.2 Học Có Giám Sát (Supervised Learning)

### 2.2.1 Định Nghĩa

**Học có giám sát** là phương pháp trong đó mô hình học từ một tập dữ liệu đã được gán nhãn. Mỗi đầu vào $x_i$ có một nhãn đầu ra $y_i$ tương ứng. Mô hình học từ dữ liệu này để xây dựng một hàm ánh xạ $f(x)$ từ đầu vào $x$ đến đầu ra $y$.

Mục tiêu của học có giám sát là xây dựng một mô hình mà khi đưa vào dữ liệu mới, mô hình có thể dự đoán chính xác giá trị đầu ra.

### 2.2.2 Các Phương Pháp Phổ Biến

1. **Hồi Quy (Regression)**: Dự đoán giá trị liên tục của một biến số. Ví dụ: Dự đoán giá trị nhà từ diện tích và số phòng.
   
   - **Hồi quy tuyến tính (Linear Regression)** là mô hình đơn giản và phổ biến.
   
     Công thức hồi quy tuyến tính:
      $$
      y = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b
      $$
     Trong đó:
     - $w_1, w_2, ..., w_n$ là các trọng số (weights),
     - $b$ là hệ số chệch (bias),
     - $x_1, x_2, ..., x_n$ là các đặc trưng đầu vào.

   - **Hồi quy phi tuyến (Non-linear Regression)**: Áp dụng cho các dữ liệu có mối quan hệ phi tuyến.

2. **Phân Loại (Classification)**: Dự đoán một nhãn hoặc nhóm cho các đối tượng. Ví dụ: Phân loại email là spam hay không spam.

   - **Mô hình phân loại tuyến tính (Linear Classification)**: Sử dụng hồi quy tuyến tính nhưng với bài toán phân loại.
   
   - **SVM (Support Vector Machine)**: Là một phương pháp phân loại mạnh mẽ, phân tách các lớp dữ liệu bằng một siêu phẳng (hyperplane) tối ưu.
   
     Công thức của SVM:
     $$
     f(x) = w^T x + b
     $$
     Trong đó $w$ là trọng số và $b$ là hệ số chệch. Phân loại được xác định qua việc tìm siêu phẳng tối ưu phân tách các lớp.

3. **K-nearest Neighbors (K-NN)**: Phương pháp phân loại hoặc hồi quy dựa trên việc tìm k điểm gần nhất trong không gian đặc trưng và dựa trên nhãn của chúng để đưa ra dự đoán.

   - Công thức tính khoảng cách Euclidean giữa hai điểm $x_i$ và $x_j$ là:
     $$
     d(x_i, x_j) = \sqrt{\sum_{k=1}^{n} (x_{ik} - x_{jk})^2}
     $$

4. **Thuật toán cây quyết định (Decision Trees)**: Dựa trên các quyết định phân nhánh để phân loại hoặc dự đoán giá trị liên tục.

   - Các cây quyết định sử dụng thuật toán **CART (Classification and Regression Tree)** để phân chia dữ liệu tại mỗi nút, lựa chọn đặc trưng tối ưu.

---

## 2.3 Học Không Giám Sát (Unsupervised Learning)

### 2.3.1 Định Nghĩa

**Học không giám sát** là phương pháp trong đó mô hình học từ dữ liệu mà không có nhãn. Mục tiêu của phương pháp này là phát hiện các cấu trúc hoặc mẫu dữ liệu tiềm ẩn mà không có sự chỉ dẫn từ nhãn đầu ra.

### 2.3.2 Các Phương Pháp Phổ Biến

1. **Phân Cụm (Clustering)**: Mục tiêu là nhóm các đối tượng có đặc trưng tương tự vào các nhóm (clusters).

   - **K-means clustering** là thuật toán phổ biến nhất để phân cụm.
   
     - Quá trình hoạt động:
       1. Chọn số cụm $K$.
       2. Khởi tạo $K$ centroid ngẫu nhiên.
       3. Gán mỗi điểm vào cụm gần nhất.
       4. Cập nhật centroid theo trung bình các điểm trong mỗi cụm.
   
     Công thức tính khoảng cách trong K-means:
     $$
     d(x_i, c_k) = \| x_i - c_k \|^2
     $$
     Trong đó $c_k$ là centroid của cụm $k$.

2. **Giảm Chiều Dữ Liệu (Dimensionality Reduction)**: Rút gọn số chiều của dữ liệu mà không làm mất quá nhiều thông tin. Một kỹ thuật phổ biến là **Phân tích thành phần chính (PCA)**.

   - **PCA** giúp tìm ra các thành phần chính, giúp giảm bớt số chiều mà vẫn giữ được hầu hết các thông tin.
   
   Phương trình PCA:
   $$
   Z = X W
   $$
   Trong đó:
   - $X$ là ma trận dữ liệu (có dạng $n \times m$),
   - $W$ là ma trận các vector riêng (eigenvectors).

3. **Mô Hình Xây Dựng (Generative Models)**: Học không giám sát cũng bao gồm các mô hình xây dựng, nơi mô hình học cách sinh dữ liệu mới từ phân phối của dữ liệu.

   - **Autoencoders** là một ví dụ phổ biến trong học sâu (deep learning), sử dụng mạng nơ-ron để học cách mã hóa và giải mã dữ liệu.

---

## 2.4 Học Tăng Cường (Reinforcement Learning)

### 2.4.1 Định Nghĩa

**Học tăng cường** là phương pháp học trong đó một tác nhân (agent) học cách hành động trong môi trường (environment) để tối đa hóa tổng phần thưởng (reward) mà nó nhận được. Không giống như học có giám sát, tác nhân không được cung cấp nhãn cho mỗi hành động mà nó thực hiện. Thay vào đó, tác nhân sẽ học từ các phản hồi (feedback) trong môi trường sau mỗi hành động.

### 2.4.2 Các Thành Phần Chính

1. **Tác Nhân (Agent)**: Là đối tượng thực hiện các hành động trong môi trường.
2. **Môi Trường (Environment)**: Là nơi tác nhân hoạt động.
3. **Phần Thưởng (Reward)**: Đánh giá của môi trường đối với hành động của tác nhân.
4. **Hành Động (Action)**: Các quyết định mà tác nhân thực hiện.

### 2.4.3 Công Thức Q-Learning

Q-learning là một thuật toán học tăng cường mạnh mẽ, trong đó tác nhân học cách tối ưu hóa hành động thông qua một bảng giá trị $Q(s, a)$, đại diện cho phần thưởng tối đa mà tác nhân có thể nhận được từ trạng thái $s$ khi thực hiện hành động $a$.

Cập nhật giá trị Q theo công thức:
$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]
$$
Trong đó:
- $Q(s_t, a_t)$ là giá trị của hành động $a_t$ tại trạng thái $s_t$,
- $\alpha$ là tốc độ học (learning rate),
- $\gamma$ là yếu tố giảm giá (discount factor),
- $r_{t+1}$ là phần thưởng nhận được sau hành động $a_t$.

---

## 2.5 Các Ứng Dụng Học Máy

Mỗi phương pháp học máy đều có những ứng dụng riêng trong thực tế:

- **Học có giám sát**: Phân loại hình ảnh, dự đoán giá trị bất động sản, phân loại bệnh tật, nhận diện giọng nói, phân tích tài chính.
- **Học không giám sát**: Phân nhóm khách hàng, phân tích dữ liệu lớn, phát hiện gian lận.
- **Học tăng cường**: Xe tự lái, trò chơi máy tính, robot tự động, tối ưu hóa chiến lược marketing.

---

## 2.6 Kết Luận

Mỗi phương pháp học máy có những đặc điểm và ứng dụng riêng.

Việc lựa chọn phương pháp nào để giải quyết bài toán phụ thuộc vào loại dữ liệu có sẵn, mục tiêu của bài toán và đặc điểm của môi trường học tập. Học máy là một lĩnh vực rộng lớn và đa dạng, vì vậy hiểu và áp dụng đúng phương pháp học máy là một yếu tố quan trọng trong quá trình phát triển các mô hình AI mạnh mẽ.