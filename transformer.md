**Kiến trúc Transformer** là một trong những mô hình học sâu mạnh mẽ nhất hiện nay, đặc biệt nổi bật trong các bài toán xử lý ngôn ngữ tự nhiên như dịch máy, tạo văn bản, và nhiều ứng dụng khác. Mô hình này được giới thiệu trong bài báo **"Attention is All You Need"** của Vaswani et al. (2017). Kiến trúc Transformer sử dụng cơ chế **self-attention** (chú ý bản thân) để xử lý thông tin đầu vào mà không cần đến cấu trúc chuỗi tuần tự như trong các mô hình RNN (Recurrent Neural Networks) hay LSTM (Long Short-Term Memory).

Dưới đây là thông tin chi tiết về kiến trúc Transformer, bao gồm các lớp, đầu vào, đầu ra, và các ký hiệu toán học liên quan.

---

### **1. Kiến Trúc Của Transformer**
Kiến trúc Transformer bao gồm hai phần chính: **Encoder** và **Decoder**, mỗi phần có thể có nhiều lớp lặp lại.

#### **A. Encoder**

Encoder trong Transformer bao gồm nhiều lớp (layers), mỗi lớp có hai thành phần chính:
1. **Self-Attention**: Giúp mô hình học cách tập trung vào các phần khác nhau trong chuỗi đầu vào.
2. **Feed Forward Neural Network**: Một mạng nơ-ron cơ bản giúp mô hình học được các sự chuyển đổi phức tạp giữa các trạng thái.

Mỗi lớp Encoder bao gồm:
- **Multi-Head Self-Attention Mechanism**
- **Position-wise Feed-Forward Networks**
- **Residual Connections và Layer Normalization**

##### **Cấu trúc của mỗi lớp Encoder**:
Giả sử ta có một chuỗi đầu vào **X** có chiều dài là **n** và mỗi từ có **d\_model** chiều. 

**Input**: 
- Chuỗi đầu vào $X = [x_1, x_2, ..., x_n]$, với mỗi $x_i \in \mathbb{R}^{d_{model}}$

**Lớp 1: Multi-Head Self-Attention**

- Đầu vào: $Q = XW_Q$, $K = XW_K$, $V = XW_V$, trong đó $W_Q, W_K, W_V$ là các ma trận trọng số học được.
- Cơ chế chú ý tính toán:
$$
  Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
  Trong đó:
  - $Q$ là **Query** (truy vấn)
  - $K$ là **Key** (chìa khóa)
  - $V$ là **Value** (giá trị)
  - $d_k$ là chiều dài của vector **Key**.
  
- **Multi-head attention** là việc chia $Q$, $K$, và $V$ thành $h$ phần (với mỗi phần có kích thước nhỏ hơn) và tính toán **Attention** độc lập cho mỗi phần, sau đó kết hợp các kết quả.
$$
  MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h)W^O
$$
  Trong đó $W^O$ là ma trận trọng số.

**Lớp 2: Position-wise Feed Forward Network**

Sau khi tính toán Attention, đầu ra được đưa qua một mạng nơ-ron feedforward, gồm hai lớp tuyến tính:
$$
FF(x) = max(0, xW_1 + b_1)W_2 + b_2
$$
- $W_1$, $W_2$, $b_1$, và $b_2$ là các ma trận trọng số và bias.
- Hàm $max(0, \cdot)$ là hàm ReLU.

**Lớp Normalize và Residual Connection**: 
Sau mỗi bước, kết quả của Attention và Feed-Forward Network được đưa vào một lớp **Layer Normalization** và cộng với **Residual Connection**:
$$
\text{Output}_l = \text{LayerNorm}(x_l + \text{SubLayer}(x_l))
$$
Trong đó **SubLayer** có thể là Attention hoặc Feed-Forward Network, tùy theo vị trí trong lớp.

---

#### **B. Decoder**

Decoder trong Transformer tương tự như Encoder nhưng có thêm một cơ chế **Encoder-Decoder Attention**, giúp decoder học được thông tin từ đầu ra của encoder. Cấu trúc của một lớp Decoder bao gồm:
1. **Masked Multi-Head Self-Attention**: Tương tự như Encoder nhưng có thêm điều kiện "masked" để đảm bảo tính tuần tự trong việc sinh ra các từ.
2. **Encoder-Decoder Attention**: Cơ chế này giúp decoder sử dụng thông tin từ encoder.
3. **Feed Forward Neural Network**: Tương tự như ở Encoder.

##### **Cấu trúc của mỗi lớp Decoder**:
**Input**: 
- Chuỗi đầu vào $Y = [y_1, y_2, ..., y_n]$ từ decoder trước đó.

**Lớp 1: Masked Multi-Head Self-Attention**

- Đây là cơ chế Attention giống như trong Encoder nhưng với một "mask" để đảm bảo rằng mỗi từ trong chuỗi decoder chỉ có thể chú ý đến các từ trước nó trong chuỗi (để duy trì tính tuần tự).

**Lớp 2: Encoder-Decoder Attention**

- Mục đích của Encoder-Decoder Attention là sử dụng thông tin từ phần encoder:
$$
  Attention_{Enc-Dec}(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
  Trong đó:
  - $Q$ đến từ Decoder.
  - $K$, $V$ đến từ Encoder.

**Lớp 3: Position-wise Feed Forward Network**

- Cũng tương tự như trong Encoder.

---

### **2. Tính Toán Đầu Vào và Đầu Ra**

- **Đầu vào (Input)**: Đầu vào cho cả Encoder và Decoder là một chuỗi các vector $X = [x_1, x_2, ..., x_n]$, mỗi $x_i \in \mathbb{R}^{d_{model}}$. Trước khi đưa vào mô hình, đầu vào sẽ trải qua một bước mã hóa vị trí (Positional Encoding), giúp mô hình nhận biết vị trí của từng từ trong chuỗi.

- **Đầu ra (Output)**: Đầu ra của mỗi lớp Encoder là một biểu diễn nâng cao của chuỗi đầu vào, và đầu ra của Decoder là một chuỗi được sinh ra (ví dụ: dịch văn bản).

---

### **3. Tổng Quan Về Mô Hình**

- **Encoder** và **Decoder** có thể được lặp lại nhiều lần (thường là 6 lớp cho mỗi phần trong bài báo gốc), giúp mô hình học các biểu diễn phức tạp hơn.
- Cả Encoder và Decoder đều sử dụng **Layer Normalization** và **Residual Connections** giúp ổn định quá trình huấn luyện và tăng khả năng học của mô hình.
- Các lớp trong Transformer có thể được song song hóa (parallelized) dễ dàng hơn so với RNN/LSTM, giúp tăng tốc độ huấn luyện.

---

### **4. Tổng Quan Các Phần Quan Trọng**
- **Self-Attention** giúp mô hình có thể học được mối quan hệ giữa các từ trong chuỗi, bất kể vị trí của chúng.
- **Multi-Head Attention** giúp mô hình tìm kiếm nhiều mối quan hệ khác nhau đồng thời.
- **Positional Encoding** giúp mô hình hiểu được thứ tự của các từ trong chuỗi.
- **Feed-Forward Networks** giúp mô hình học các tính toán phức tạp hơn.

---

### **5. Kết Luận**

Kiến trúc Transformer, với cơ chế Attention là yếu tố chính giúp mô hình này vượt trội so với các mô hình học sâu khác trong các bài toán xử lý ngôn ngữ tự nhiên. Nhờ vào việc loại bỏ các yếu tố tuần tự như trong RNN/LSTM, Transformer có thể xử lý các chuỗi dữ liệu một cách hiệu quả và dễ dàng song song hóa, giúp giảm thời gian huấn luyện và cải thiện hiệu suất.

---

### **Các Tài Liệu Tham Khảo**:
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A., Kaiser, Ł., Polosukhin, I. (2017). "Attention is All You Need". NIPS 2017.

