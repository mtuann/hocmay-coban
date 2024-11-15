Dưới đây là nội dung chi tiết và đầy đủ cho **Chương 1: Giới thiệu về Học Máy (Machine Learning)**, bao gồm các khái niệm cơ bản, phân loại các phương pháp học máy, công thức toán học và các ứng dụng phổ biến của học máy.

---

# Chương 1: Giới Thiệu Về Học Máy (Machine Learning)

## 1.1 Học Máy Là Gì?

**Học máy (Machine Learning - ML)** là một lĩnh vực của trí tuệ nhân tạo (AI) nghiên cứu và phát triển các thuật toán giúp máy tính có thể học từ dữ liệu và tự động cải thiện khả năng dự đoán hoặc quyết định mà không cần phải lập trình chi tiết cho mỗi trường hợp cụ thể. Học máy sử dụng các mô hình toán học để nhận diện các mẫu dữ liệu và đưa ra dự đoán hoặc quyết định dựa trên những mẫu đó.

Một mô hình học máy có thể được huấn luyện với một tập dữ liệu (training data) và từ đó dự đoán các giá trị trên các dữ liệu chưa thấy trước đó (test data).

### Các Đặc Điểm Chính Của Học Máy:

- **Học từ dữ liệu**: Học máy là quá trình mà máy tính học cách nhận diện và phân tích dữ liệu, từ đó đưa ra các dự đoán hoặc quyết định.
- **Cải thiện qua thời gian**: Khi có thêm dữ liệu hoặc điều chỉnh mô hình, độ chính xác của mô hình sẽ được cải thiện.
- **Tự động hóa quyết định**: Máy tính có thể tự động đưa ra quyết định hoặc dự đoán dựa trên các mẫu học được mà không cần sự can thiệp của con người.

---

## 1.2 Các Phương Pháp Học Máy

Học máy có thể được phân loại thành các nhóm phương pháp khác nhau dựa trên cách thức học và dữ liệu mà chúng sử dụng. Ba nhóm phương pháp chính là: **Học có giám sát**, **Học không giám sát** và **Học tăng cường**.

### 1.2.1 Học Có Giám Sát (Supervised Learning)

**Học có giám sát** là phương pháp học máy trong đó mô hình học từ một tập dữ liệu đã được gán nhãn (labeled data). Dữ liệu đầu vào (input) sẽ có nhãn đầu ra (output) tương ứng, và mục tiêu của mô hình là học cách ánh xạ đầu vào tới đầu ra sao cho khi gặp dữ liệu mới, mô hình có thể dự đoán chính xác đầu ra.

#### Công thức và Toán học cơ bản:

Trong học có giám sát, mô hình học một hàm $h: X \to Y$, trong đó:
- $X$ là không gian đầu vào (input space),
- $Y$ là không gian đầu ra (output space).

Các dạng bài toán trong học có giám sát:
- **Phân loại (Classification)**: Dự đoán nhãn cho một đối tượng. Ví dụ: phân loại email là spam hoặc không spam.
- **Hồi quy (Regression)**: Dự đoán giá trị liên tục cho một đối tượng. Ví dụ: dự đoán giá trị nhà từ diện tích và số phòng.

**Công thức hồi quy tuyến tính (Linear Regression)**:
Hàm hồi quy tuyến tính mô tả một mối quan hệ giữa biến phụ thuộc $y$ và biến độc lập $x$ theo dạng:
$$
y = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b
$$
Trong đó:
- $w_1, w_2, ..., w_n$ là các trọng số (weights),
- $b$ là hệ số chệch (bias),
- $x_1, x_2, ..., x_n$ là các đặc trưng của dữ liệu.

Mục tiêu trong học có giám sát là tối ưu hóa các trọng số $w$ và $b$ sao cho mô hình dự đoán $\hat{y}$ có sai số nhỏ nhất so với giá trị thực tế $y$.

#### Ví dụ:
- **Phân loại**: Dự đoán một email có phải là spam hay không.
- **Hồi quy**: Dự đoán giá trị của một căn nhà dựa trên các đặc trưng như diện tích, số phòng ngủ.

---

### 1.2.2 Học Không Giám Sát (Unsupervised Learning)

**Học không giám sát** là phương pháp học máy trong đó dữ liệu huấn luyện không có nhãn (labels). Mục tiêu của học không giám sát là khám phá ra các cấu trúc, mối quan hệ tiềm ẩn trong dữ liệu mà không cần bất kỳ thông tin về nhãn đầu ra.

#### Các kỹ thuật phổ biến trong học không giám sát:
- **Phân cụm (Clustering)**: Nhóm các đối tượng tương tự vào cùng một nhóm. Một thuật toán phổ biến là **K-means**.
  
  Công thức phân cụm K-means:
  1. Khởi tạo $K$ centroid ngẫu nhiên.
  2. Phân nhóm các điểm dữ liệu theo khoảng cách tới các centroid gần nhất:
  $$
  C_i = \arg \min_k \| x_i - c_k \|^2
  $$
  3. Cập nhật lại các centroid dựa trên trung bình của các điểm trong nhóm:
  $$
  c_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i
  $$
- **Giảm chiều dữ liệu (Dimensionality Reduction)**: Rút gọn số chiều của dữ liệu trong khi vẫn giữ lại các thông tin quan trọng. Một kỹ thuật phổ biến là **Phân tích thành phần chính (PCA)**.

  Phương trình PCA:
  $$
  Z = X W
  $$
  Trong đó:
  - $X$ là ma trận dữ liệu đầu vào (dạng $n \times m$, với $n$ là số lượng điểm dữ liệu và $m$ là số đặc trưng),
  - $W$ là ma trận các vector riêng (eigenvectors) của ma trận hiệp phương sai.

#### Ví dụ:
- **Phân cụm**: Nhóm các khách hàng có hành vi mua sắm tương tự.
- **Giảm chiều dữ liệu**: Rút gọn dữ liệu về hình ảnh hoặc văn bản để sử dụng hiệu quả hơn trong các mô hình học máy.

---

### 1.2.3 Học Tăng Cường (Reinforcement Learning)

**Học tăng cường** là phương pháp học máy trong đó một tác nhân (agent) học cách thực hiện các hành động trong một môi trường để tối đa hóa phần thưởng (reward). Mô hình không nhận được nhãn cụ thể cho mỗi hành động mà phải tự học từ trải nghiệm qua việc thử và sai (trial and error).

#### Các thành phần cơ bản:
- **Tác nhân (Agent)**: Thực hiện hành động trong môi trường.
- **Môi trường (Environment)**: Nơi mà tác nhân hoạt động.
- **Hành động (Action)**: Các hành động mà tác nhân có thể thực hiện.
- **Phần thưởng (Reward)**: Đánh giá của môi trường đối với hành động của tác nhân.

Công thức cơ bản của học tăng cường:
$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]
$$
Trong đó:
- $Q(s_t, a_t)$ là giá trị của hành động $a_t$ tại trạng thái $s_t$,
- $r_{t+1}$ là phần thưởng nhận được sau khi thực hiện hành động $a_t$,
- $\gamma$ là yếu tố giảm giá (discount factor),
- $\alpha$ là tốc độ học (learning rate).

#### Ví dụ:
- **Chơi game**: Tác nhân học cách chơi cờ vua, chơi game Pac-Man, hoặc điều khiển robot qua các nhiệm vụ thử thách.

---

## 1.3 Các Ứng Dụng Của Học Máy

Học máy hiện nay đã có mặt trong rất nhiều lĩnh vực và có tác động lớn đến đời sống. Một số ứng dụng phổ biến bao gồm:

- **Phân loại ảnh và nhận diện hình ảnh**: Học máy giúp nhận diện các đối tượng trong ảnh, ví dụ như nhận diện khuôn mặt, phân loại động vật trong ảnh, v.v.
- **Phân tích ngữ nghĩa trong văn bản**: Các mô hình học máy như xử lý ngôn ngữ tự nhiên (NLP) giúp hiểu và phân tích văn bản, ví dụ: phân tích cảm xúc, dịch máy, phân loại văn bản.
- **Dự đoán tài chính**: Học máy được sử dụng trong các mô hình dự đoán thị trường chứng khoán, phân tích hành vi khách hàng trong ngân hàng, dự đoán giá cổ phiếu.
- **Xe tự lái**: Học máy đóng vai trò quan trọng trong việc giúp xe tự lái nhận diện và điều hướng giao thông.

---

## 1.4 Kết Luận

Học máy là một lĩnh vực quan trọng và đầy tiềm năng, với ứng dụng rộng

 rãi trong nhiều ngành nghề. Hiểu rõ các phương pháp và ứng dụng cơ bản của học máy là bước đầu tiên quan trọng để các bạn có thể tiếp tục nghiên cứu và phát triển các mô hình học máy phức tạp hơn trong tương lai.