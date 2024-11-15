### **Tokenization và Embeddings trong Học Máy**

Trong các mô hình học máy xử lý ngôn ngữ tự nhiên (NLP), **Tokenization** và **Embeddings** là hai khái niệm cực kỳ quan trọng. Cả hai đều đóng vai trò trong việc chuyển đổi văn bản đầu vào thành các dạng có thể xử lý được bởi các mô hình học máy.

### **1. Tokenization**

**Tokenization** là quá trình phân tách văn bản thành các đơn vị nhỏ hơn, gọi là **tokens**. Một token có thể là một từ, một ký tự, hoặc một phần của từ (ví dụ: các từ vựng, dấu câu hoặc thậm chí các chữ cái hoặc âm tiết trong từ).

#### **Quy trình Tokenization**

- **Word-level tokenization**: Tách văn bản thành các từ.
  Ví dụ: Câu "I love machine learning" sẽ được token hóa thành các tokens: `["I", "love", "machine", "learning"]`.
  
- **Subword-level tokenization**: Tách văn bản thành các đơn vị con của từ, như các ký tự hoặc phần của từ. Kỹ thuật này hữu ích trong các mô hình như **Byte Pair Encoding (BPE)** hoặc **WordPiece**. Ví dụ, từ "unhappiness" có thể được tách thành `["un", "##happiness"]`.

- **Character-level tokenization**: Tách văn bản thành các ký tự riêng lẻ.
  Ví dụ: Câu "Hi" sẽ được token hóa thành `["H", "i"]`.

- **Sentence-level tokenization**: Tách văn bản thành các câu.
  Ví dụ: Câu "I love machine learning. It's amazing!" sẽ được token hóa thành `["I love machine learning.", "It's amazing!"]`.

#### **Ví dụ về Tokenization**

Câu: **"Tokenization is fun!"**
- Word-level tokenization: `["Tokenization", "is", "fun", "!"]`
- Subword-level tokenization (sử dụng BPE): `["Token", "ization", "is", "fun", "!"]`
- Character-level tokenization: `["T", "o", "k", "e", "n", "i", "z", "a", "t", "i", "o", "n", " ", "i", "s", " ", "f", "u", "n", "!"]`

**Tokenization** giúp chuyển văn bản thành các đơn vị mà máy tính có thể xử lý. Tuy nhiên, với mỗi đơn vị này, chúng ta cần phải chuyển đổi chúng thành các vector số học (mảng số) để đưa vào mô hình học máy. Đây là nơi **Embeddings** xuất hiện.

### **2. Embeddings**

**Embeddings** là quá trình chuyển đổi các tokens (từ hoặc các đơn vị con của từ) thành các vector số có độ chiều thấp. Các vector này giúp máy tính có thể "hiểu" được mối quan hệ giữa các từ, từ đó cải thiện hiệu quả trong các tác vụ xử lý ngôn ngữ tự nhiên (NLP).

#### **Cách thức hoạt động của Embeddings**

Embeddings thực chất là việc ánh xạ các tokens (thường là từ trong ngữ cảnh NLP) vào một không gian vector, nơi mà các từ có nghĩa tương tự sẽ có vector gần nhau.

Ví dụ, từ "cat" và "dog" có thể có các vector gần nhau trong không gian embedding vì chúng có ý nghĩa tương tự trong ngữ cảnh.

Các phương pháp phổ biến để tạo ra **word embeddings** bao gồm:
- **Word2Vec**
- **GloVe (Global Vectors for Word Representation)**
- **FastText**
- **BERT** (cung cấp embedding context-sensitive, nghĩa là từ "bank" sẽ có embedding khác khi xuất hiện trong "river bank" và "bank account")

#### **Công Thức Cơ Bản của Word Embeddings**

Word embeddings thường được huấn luyện với các thuật toán như **Skip-Gram** hoặc **CBOW (Continuous Bag of Words)** trong mô hình **Word2Vec**.

- **Skip-Gram**: Dự đoán các từ trong ngữ cảnh (context) dựa trên một từ mục tiêu.
$$
  P(\text{context} | \text{target}) = \prod_{i=1}^{C} P(w_i | w_{\text{target}})
$$
  Trong đó $C$ là kích thước cửa sổ ngữ cảnh, và $w_i$ là các từ trong cửa sổ ngữ cảnh của từ mục tiêu.

- **CBOW**: Dự đoán từ mục tiêu từ các từ trong ngữ cảnh.
$$
  P(w_{\text{target}} | \text{context}) = \prod_{i=1}^{C} P(w_{\text{target}} | w_i)
$$

Các từ trong mô hình Word2Vec được ánh xạ thành các vector trong không gian **d-dimensional** (ví dụ: 100, 300 hoặc 500 chiều). Các từ có cùng ý nghĩa hoặc xuất hiện trong ngữ cảnh tương tự sẽ có vector gần nhau.

#### **GloVe Embeddings**

GloVe là một phương pháp khác để tạo ra word embeddings, dựa trên thống kê tổng hợp từ các ma trận tương quan của từ vựng trong một tập dữ liệu lớn.

Công thức GloVe:
$$
J = \sum_{i,j=1}^{V} f(X_{ij}) \left( \mathbf{w_i}^T \mathbf{w_j} + b_i + b_j - \log X_{ij} \right)^2
$$
Trong đó:
- $X_{ij}$ là số lần từ $i$ xuất hiện trong ngữ cảnh của từ $j$,
- $\mathbf{w_i}$ và $\mathbf{w_j}$ là embedding vector của các từ $i$ và $j$,
- $b_i$ và $b_j$ là bias terms,
- $f(X_{ij})$ là một hàm điều chỉnh (thường là một hàm xếp hạng tần suất).

#### **Embedding Lookup Table**

Khi huấn luyện mô hình, mỗi từ trong từ điển (vocabulary) sẽ được ánh xạ với một vector embedding. Những vectors này thường được lưu trong một **embedding matrix** $E$, với mỗi hàng của ma trận là vector embedding của một từ.

Giả sử $V$ là số lượng từ trong từ điển, và $d$ là kích thước của vector embedding, thì ma trận embedding $E$ có kích thước $V \times d$. Mỗi vector $\mathbf{v_i}$ (vector embedding của từ $w_i$) có thể được tra cứu từ ma trận embedding $E$.

$$
\mathbf{v_i} = E[w_i]
$$

#### **Contextual Embeddings**

Mặc dù các phương pháp như **Word2Vec** và **GloVe** cung cấp các embedding từ cố định cho mỗi từ, nhưng các mô hình mới như **BERT** và **GPT** tạo ra **contextual embeddings**. Điều này có nghĩa là mỗi từ có thể có nhiều embedding khác nhau tuỳ vào ngữ cảnh mà nó xuất hiện.

Ví dụ:
- Từ **"bank"** trong "river bank" và "bank account" sẽ có embeddings khác nhau trong **BERT** vì ngữ cảnh của chúng khác nhau.

### **Ví Dụ Cụ Thể về Embedding**

Giả sử ta có một từ vựng với các từ **["cat", "dog", "apple", "orange"]** và một embedding không gian 2 chiều (2D).

- "cat" có thể có vector embedding: $[0.2, 0.4]$
- "dog" có thể có vector embedding: $[0.3, 0.5]$
- "apple" có thể có vector embedding: $[0.9, 0.8]$
- "orange" có thể có vector embedding: $[0.8, 0.9]$

Trong không gian này, ta có thể thấy rằng từ **"cat"** và **"dog"** có vector gần nhau vì chúng có nghĩa gần nhau, trong khi **"apple"** và **"orange"** cũng có vector gần nhau do chúng đều là các loại quả.

### **Tóm Tắt**

- **Tokenization** là quá trình phân tách văn bản thành các đơn vị nhỏ hơn (tokens), có thể là từ, ký tự hoặc phần của từ.
- **Word Embeddings** là quá trình ánh xạ các từ thành các vector trong không gian liên tục, giúp mô hình hiểu được mối quan hệ giữa các từ.
- Các phương pháp phổ biến để huấn luyện word embeddings bao gồm **Word2Vec**, **GloVe**, và các mô hình **contextual embeddings** như **BERT**.
