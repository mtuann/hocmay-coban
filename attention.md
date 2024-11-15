### **Attention Mechanism trong Mạng Neural**

**Attention Mechanism** là một cơ chế trong các mô hình học sâu, đặc biệt là trong các mô hình chuỗi như Transformer, giúp mô hình "chú ý" đến các phần quan trọng của đầu vào khi xử lý một phần dữ liệu cụ thể. Điều này rất hữu ích trong các nhiệm vụ như dịch máy, tóm tắt văn bản, hay nhận dạng đối tượng trong hình ảnh.

#### **Khái Niệm Cơ Bản**

Cơ chế attention giúp mô hình xác định **mối quan hệ** giữa các phần khác nhau trong đầu vào (như các từ trong câu) để tạo ra một đại diện chính xác hơn cho từng phần của đầu ra. Cụ thể, đối với một từ (token) trong một chuỗi đầu vào, attention tính toán "chú ý" của từ đó đối với tất cả các từ khác trong chuỗi đầu vào, với mục tiêu trọng số hóa các phần có ảnh hưởng lớn hơn tới kết quả đầu ra.

### **Các Thành Phần trong Attention Mechanism**

Cơ chế attention sử dụng ba thành phần cơ bản: **Query (Q), Key (K), Value (V)**.

- **Query (Q)**: Là đại diện của phần cần chú ý (ví dụ: từ hiện tại trong quá trình dịch máy).
- **Key (K)**: Là đại diện của các phần trong đầu vào mà mô hình sẽ so sánh với query.
- **Value (V)**: Là giá trị mà mô hình sử dụng để tạo ra đầu ra.

### **Tính Toán Attention**

Công thức cơ bản của attention có thể được mô tả qua các bước sau:

1. **Tính similarity giữa Query và Key**: 

   Để tính mức độ "chú ý" giữa Query và Key, ta sử dụng một hàm similarity. Một trong những cách phổ biến là sử dụng **product vô hướng (dot product)** giữa Query và Key:

$$
\text{Score}(Q, K) = Q^T K
$$

   Kết quả của phép toán này là một điểm số thể hiện mức độ tương đồng giữa query và key. Nếu điểm số này cao, điều đó có nghĩa là phần tử đầu vào có ảnh hưởng lớn hơn đến phần tử hiện tại trong chuỗi.

2. **Chuyển điểm số thành xác suất** (Softmax):

   Điểm số thu được ở bước trên sẽ được đưa vào hàm **softmax** để chuyển đổi thành xác suất, đảm bảo tổng của các trọng số attention là 1. Điều này giúp mô hình "tập trung" vào những phần quan trọng nhất:

   $$
   \alpha_{i} = \frac{ \exp(\text{Score}(Q, K_i)) }{ \sum_j \exp(\text{Score}(Q, K_j)) }
   $$

   Trong đó, $\alpha_{i}$ là trọng số attention cho phần tử thứ $i$, $K_i$ là key tương ứng với phần tử thứ $i$.

3. **Trọng số hóa Value**:

   Trọng số này được sử dụng để tính toán giá trị đầu ra, bằng cách nhân nó với Value (V) tương ứng với mỗi Key:

   $$
   \text{Output} = \sum_{i} \alpha_{i} V_i
   $$

   Kết quả này là đầu ra của phần tử query, dựa trên mức độ chú ý vào các phần tử trong chuỗi đầu vào.

### **Cơ Chế Attention Cơ Bản - Quy Trình**

Tóm lại, quy trình tính toán attention cho một **Query** duy nhất có thể được mô tả qua ba bước chính:

1. Tính điểm số tương quan giữa Query và mỗi Key, thường sử dụng phép nhân vô hướng (dot product).
2. Áp dụng hàm softmax lên các điểm số để nhận các trọng số attention.
3. Nhân các trọng số với các Value tương ứng và tính tổng để nhận đầu ra.

### **Scaled Dot-Product Attention**

Một phiên bản cải tiến của Attention là **Scaled Dot-Product Attention**, được sử dụng trong các mô hình Transformer. Trong phương pháp này, thay vì tính trực tiếp sản phẩm vô hướng giữa Query và Key, chúng ta chia kết quả đó cho căn bậc hai của chiều dài của các vector (được gọi là **d_k**) để tránh vấn đề gradient vanishing khi các sản phẩm vô hướng có giá trị quá lớn:

$$
\text{Score}(Q, K) = \frac{Q^T K}{\sqrt{d_k}}
$$

<!-- $$
\text{Score}(Q, K) = Q^T K
$$

\[
\text{Score}(Q, K) = Q^T K
\] -->


Ở đây, $d_k$ là chiều dài của vector Key (hoặc Query), và $\sqrt{d_k}$ là bước chia để chuẩn hóa.

### **Multi-Head Attention**

**Multi-Head Attention** là một cải tiến quan trọng của cơ chế attention trong Transformer. Thay vì sử dụng chỉ một cặp Query, Key, Value, mô hình sử dụng nhiều bộ attention song song (gọi là heads). Điều này giúp mô hình có thể học được nhiều mối quan hệ khác nhau giữa các phần trong dữ liệu.

Quy trình của Multi-Head Attention có thể tóm tắt như sau:

1. Chia các vector Query, Key, Value thành nhiều phần nhỏ (head).
2. Tính toán attention độc lập cho mỗi head, sử dụng các công thức đã trình bày.
3. Kết hợp các đầu ra của từng head lại thành một vector duy nhất.

Công thức cho Multi-Head Attention:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h) W^O
$$

Trong đó:

- $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$
- $W^O$ là ma trận trọng số để chuyển đổi kết quả cuối cùng.

### **Self-Attention**

**Self-Attention** là một dạng đặc biệt của attention, nơi **Query, Key, và Value** đều đến từ cùng một nguồn (ví dụ: cùng một chuỗi đầu vào). Điều này có nghĩa là mô hình có thể chú ý đến các phần khác nhau trong chính văn bản đầu vào của nó khi xử lý một từ (token) nhất định.

Quy trình của **Self-Attention** trong một chuỗi văn bản có thể được mô tả như sau:

- Với mỗi từ trong câu, ta sẽ tính toán sự tương quan (similarity) giữa từ đó với tất cả các từ còn lại trong câu.
- Mỗi từ sẽ có một trọng số tương ứng cho mỗi từ khác, cho biết mức độ ảnh hưởng của các từ đó đối với từ hiện tại.

### **Công Thức Tổng Quát cho Attention Mechanism**

Công thức tổng quát cho một lớp Attention (có thể là Self-Attention) có thể được mô tả như sau:

$$
\text{Attention}(Q, K, V) = \text{Softmax} \left( \frac{Q K^T}{\sqrt{d_k}} \right) V
$$

- $Q$, $K$, và $V$ là các ma trận của Query, Key, và Value.
- $d_k$ là chiều dài của vector Key.
- $\frac{Q K^T}{\sqrt{d_k}}$ là điểm số của sự chú ý được chuẩn hóa.
- $\text{Softmax}$ giúp chuyển điểm số thành xác suất, tạo ra các trọng số attention.
- $V$ là ma trận Value sẽ được trọng số hóa và tổng hợp để tạo đầu ra.

### **Cấu Trúc Transformer**

Trong Transformer, cơ chế attention được sử dụng trong các **Encoder** và **Decoder**.

- **Encoder** sử dụng Self-Attention để xử lý đầu vào (dữ liệu), giúp mô hình hiểu mối quan hệ giữa các từ trong câu.
- **Decoder** sử dụng Self-Attention để tự chú ý vào các từ đã được sinh ra, và sử dụng **Cross-Attention** (hoặc Encoder-Decoder Attention) để chú ý đến các đầu vào từ Encoder khi sinh ra từ tiếp theo.

Cấu trúc này được tối ưu hóa cho việc xử lý các chuỗi dữ liệu dài và quan hệ dài hạn trong chuỗi.

### **Tóm Tắt**

- **Attention** là một cơ chế mạnh mẽ giúp mô hình tập trung vào những phần quan trọng nhất trong dữ liệu đầu vào.
- Các thành phần chính của attention bao gồm Query, Key, và Value.
- **Scaled Dot-Product Attention** cải tiến giúp giải quyết vấn đề liên quan đến độ lớn của các sản phẩm vô hướng.
- **Multi-Head Attention** giúp mô hình học được nhiều thông tin từ các mối quan hệ khác nhau trong đầu vào.
- **Self-Attention** là một dạng của attention, trong đó Query, Key, và Value đều đến từ đầu vào giống nhau.

Cơ chế attention này là thành phần chủ yếu trong kiến trúc **Transformer**, và là nền tảng của các mô hình học sâu hiện đại như **BERT**, **GPT**, **T5**, v.v.