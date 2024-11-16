## 1. Common Evaluation Metrics in Machine Learning
Để đánh giá hiệu quả của các mô hình học máy, đặc biệt là trong các tác vụ NLP, computer vision và học sâu (deep learning), các **evaluation metrics** (chỉ số đánh giá) đóng một vai trò quan trọng. Dưới đây là danh sách chi tiết các chỉ số đánh giá thường dùng trong học máy, cùng với cách thức tính toán và các trường hợp sử dụng phù hợp.

---

### **1. Đánh Giá Trong Phân Loại (Classification Metrics)**

#### **1.1. Accuracy (Độ Chính Xác)**
- **Định nghĩa**: Là tỷ lệ các dự đoán chính xác so với tổng số mẫu.
- **Công thức**:
$$
  \text{Accuracy} = \frac{\text{Số dự đoán chính xác}}{\text{Tổng số mẫu}}
$$
- **Áp dụng**: Phù hợp với các bài toán phân loại khi dữ liệu phân bố đều giữa các lớp.
- **Ví dụ**: Phân loại email là spam hay không spam.

#### **1.2. Precision (Độ Chính Xác)**
- **Định nghĩa**: Precision đo lường tỷ lệ các dự đoán đúng trong số các mẫu được mô hình dự đoán là dương tính (positive).
- **Công thức**:
$$
  \text{Precision} = \frac{TP}{TP + FP}
$$
  Trong đó:
  - $TP$: True Positives (Dự đoán đúng là dương tính)
  - $FP$: False Positives (Dự đoán sai là dương tính)
- **Áp dụng**: Sử dụng trong các bài toán khi chi phí của việc dự đoán sai là dương tính là cao (ví dụ: phân loại bệnh).
- **Ví dụ**: Trong phân loại bệnh, Precision cao đảm bảo rằng những người được xác nhận mắc bệnh thực sự có bệnh.

#### **1.3. Recall (Độ Nhạy)**
- **Định nghĩa**: Recall đo lường tỷ lệ mẫu dương tính thực sự được mô hình nhận diện đúng.
- **Công thức**:
$$
  \text{Recall} = \frac{TP}{TP + FN}
$$
  Trong đó:
  - $TP$: True Positives (Dự đoán đúng là dương tính)
  - $FN$: False Negatives (Dự đoán sai là âm tính)
- **Áp dụng**: Sử dụng khi chi phí của việc bỏ sót mẫu dương tính là cao.
- **Ví dụ**: Trong việc phát hiện bệnh, Recall cao đảm bảo rằng ít bệnh nhân nào bị bỏ sót.

#### **1.4. F1-Score**
- **Định nghĩa**: Là trung bình hài hòa của Precision và Recall. F1-Score giúp cân bằng giữa Precision và Recall, đặc biệt trong trường hợp các lớp không cân đối.
- **Công thức**:
$$
  F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$
- **Áp dụng**: Sử dụng khi có sự không cân bằng giữa các lớp và cần một chỉ số tổng hợp.
- **Ví dụ**: Phát hiện gian lận tín dụng.

#### **1.5. ROC-AUC**
- **Định nghĩa**: **Receiver Operating Characteristic Curve (ROC)** là một đồ thị biểu diễn mối quan hệ giữa **True Positive Rate (TPR)** và **False Positive Rate (FPR)**. **AUC** (Area Under Curve) là diện tích dưới đường cong ROC, giúp đánh giá khả năng phân biệt các lớp của mô hình.
- **Công thức**: Không có công thức cụ thể cho AUC, nhưng nó là diện tích dưới đồ thị ROC.
- **Áp dụng**: Sử dụng khi cần đánh giá khả năng phân biệt giữa các lớp trong các bài toán phân loại.
- **Ví dụ**: Phân loại email spam, phân loại bệnh.

#### **1.6. Confusion Matrix (Ma Trận Nhầm Lẫn)**
- **Định nghĩa**: Là bảng biểu diễn số lượng dự đoán đúng và sai của các mẫu trong các lớp khác nhau.
- **Công thức**: 
$$
  \begin{bmatrix}
  TP & FN \\
  FP & TN
  \end{bmatrix}
$$
  Trong đó:
  - $TP$: True Positives (Dự đoán đúng là dương tính)
  - $TN$: True Negatives (Dự đoán đúng là âm tính)
  - $FP$: False Positives (Dự đoán sai là dương tính)
  - $FN$: False Negatives (Dự đoán sai là âm tính)
- **Áp dụng**: Giúp hiểu rõ hơn về các lỗi mà mô hình gặp phải trong phân loại.

---

### **2. Đánh Giá Trong Hồi Quy (Regression Metrics)**

#### **2.1. Mean Absolute Error (MAE)**
- **Định nghĩa**: Là trung bình của sai số tuyệt đối giữa giá trị thực và giá trị dự đoán.
- **Công thức**:
$$
  \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$
  Trong đó:
  - $y_i$: Giá trị thực
  - $\hat{y}_i$: Giá trị dự đoán
- **Áp dụng**: MAE ít nhạy cảm với các ngoại lệ (outliers) so với các chỉ số khác.
- **Ví dụ**: Dự đoán giá nhà.

#### **2.2. Mean Squared Error (MSE)**
- **Định nghĩa**: Là trung bình của bình phương sai số giữa giá trị thực và giá trị dự đoán.
- **Công thức**:
$$
  \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$
- **Áp dụng**: MSE nhạy cảm với các ngoại lệ (outliers), vì bình phương sai số khiến lỗi lớn có ảnh hưởng mạnh hơn.
- **Ví dụ**: Dự đoán nhiệt độ.

#### **2.3. Root Mean Squared Error (RMSE)**
- **Định nghĩa**: Là căn bậc hai của MSE, giúp đưa kết quả về cùng đơn vị với giá trị thực.
- **Công thức**:
$$
  \text{RMSE} = \sqrt{\text{MSE}}
$$
- **Áp dụng**: RMSE giúp hiểu mức độ sai lệch giữa giá trị thực và dự đoán.
- **Ví dụ**: Dự đoán chiều cao của một nhóm người.

#### **2.4. R-squared (R²)**
- **Định nghĩa**: Là tỷ lệ biến thiên giải thích được bởi mô hình.
- **Công thức**:
$$
  R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$
  Trong đó:
  - $y_i$: Giá trị thực
  - $\hat{y}_i$: Giá trị dự đoán
  - $\bar{y}$: Giá trị trung bình của $y_i$
- **Áp dụng**: Giúp đánh giá tỷ lệ giải thích biến thiên của mô hình.

---

### **3. Đánh Giá Trong Các Tác Vụ Sinh Văn Bản (NLP)**

#### **3.1. BLEU (Bilingual Evaluation Understudy)**
- **Định nghĩa**: Là chỉ số đánh giá độ tương đồng giữa văn bản được sinh ra và văn bản tham chiếu (reference text). BLEU thường được sử dụng để đánh giá mô hình dịch máy.
- **Công thức**:
$$
  BLEU = BP \times \exp \left( \sum_{n=1}^{N} w_n \log p_n \right)
$$
  Trong đó:
  - $BP$ là **Brevity Penalty** (phạt độ dài ngắn)
  - $p_n$ là tỷ lệ trùng khớp n-gram giữa văn bản dự đoán và văn bản tham chiếu.
  - $w_n$ là trọng số cho từng cấp độ n-gram.
- **Áp dụng**: Dùng trong các bài toán dịch máy, tóm tắt văn bản.

#### **3.2. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
- **Định nghĩa**: Được sử dụng để đánh giá chất lượng văn bản sinh ra (ví dụ như tóm tắt) bằng cách so sánh với văn bản tham chiếu.
- **Công thức**: Tương tự như BLEU, ROUGE tính toán các chỉ số trùng khớp n-gram (Recall).
- **Áp dụng**: Đánh giá các mô hình tóm tắt văn bản.

---

### **4. Đánh Giá Trong Các Tác Vụ Sinh Hình Ảnh (Computer Vision Metrics)**

#### **4.1. IoU (Intersection over Union)**
- **Định nghĩa**: Được sử dụng trong các bài toán phân vùng ảnh, IoU đo lường sự trùng lặp giữa vùng dự đoán và vùng thật sự.
- **Công thức**:
$$
  IoU = \frac{\text{Khu vực giao nhau}}{\text{Khu vực hợp nhất}}
$$
- **Áp dụng**: Đánh giá các mô hình phân vùng ảnh hoặc nhận diện đối tượng.

#### **4.2. mAP (mean Average Precision)**
- **Định nghĩa**: Được sử dụng trong các bài toán phát hiện đối tượng, mAP là trung bình của các precision tại mỗi recall điểm.
- **Công thức**:
$$
  \text{mAP} = \frac{1}{N} \sum_{i=1}^{N} AP_i
$$
  Trong đó:
  - $AP_i$: Average Precision cho đối tượng i.
- **Áp dụng**: Đánh giá các mô hình nhận diện đối tượng.

---

### **Tóm tắt**

Chỉ số đánh giá (evaluation metrics) là công cụ quan trọng để giúp đánh giá và cải thiện mô hình học máy. Các chỉ số khác nhau được áp dụng cho các bài toán khác nhau, từ phân loại, hồi quy đến các tác vụ trong xử lý ngôn ngữ tự nhiên và nhận diện hình ảnh. Việc lựa chọn chỉ số phù hợp sẽ giúp mô hình của bạn đạt được hiệu quả tốt nhất tùy theo mục tiêu và bài toán nghiên cứu.


Các mô hình sinh (Generative AI), bao gồm các mô hình như **Generative Adversarial Networks (GANs)**, **Variational Autoencoders (VAEs)**, và các mô hình sinh văn bản như **GPT** hoặc **Transformer-based** models, có những đặc thù riêng biệt và yêu cầu các chỉ số đánh giá (metrics) khác nhau để đo lường chất lượng của chúng. Dưới đây là một số **evaluation metrics** quan trọng được sử dụng cho các mô hình sinh (generative models), cùng với giải thích và công thức toán học liên quan.

## 2. Evaluation Metrics for Generative Models

### **1. Inception Score (IS)**

**Định nghĩa**: Inception Score (IS) được sử dụng để đánh giá chất lượng của các hình ảnh được sinh ra từ các mô hình sinh, đặc biệt là trong các mô hình như **Generative Adversarial Networks (GANs)**. Metric này sử dụng một mô hình phân loại đã được huấn luyện trước (chẳng hạn như Inception-v3) để đo lường độ "tương tự" của hình ảnh được sinh ra với các lớp phân loại, đồng thời đánh giá độ đa dạng của các hình ảnh đó.

- **Công thức**:
$$
  IS = \exp \left( \mathbb{E}_{x \sim p_g(x)} D_K \left(p(y|x) \parallel p(y) \right) \right)
$$
  Trong đó:
  - $x$ là hình ảnh được sinh ra.
  - $p(y|x)$ là xác suất của lớp phân loại được dự đoán cho hình ảnh $x$ từ mô hình phân loại Inception.
  - $p(y)$ là phân phối xác suất của các lớp trong tập dữ liệu.
  - $D_K(p \parallel q)$ là khoảng cách Kullback-Leibler (KL Divergence) giữa hai phân phối xác suất $p$ và $q$.

**Áp dụng**: 
- Được sử dụng chủ yếu để đánh giá chất lượng của các hình ảnh được sinh ra từ mô hình GAN.
- **Ưu điểm**: Đánh giá được cả độ sắc nét (sharpness) của hình ảnh và sự đa dạng của các đối tượng trong bộ dữ liệu.
- **Nhược điểm**: Có thể không phản ánh chính xác chất lượng hình ảnh đối với các mô hình sinh không phải là hình ảnh.

---

### **2. Fréchet Inception Distance (FID)**

**Định nghĩa**: FID là một chỉ số phổ biến được sử dụng để đánh giá sự khác biệt giữa phân phối của hình ảnh sinh ra và phân phối của hình ảnh thực tế. FID được tính toán bằng cách so sánh các đặc trưng (features) của các hình ảnh được sinh ra với các đặc trưng của hình ảnh thật (sử dụng mạng phân loại Inception).

- **Công thức**:
$$
  FID = \| \mu_r - \mu_g \|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
$$
  Trong đó:
  - $\mu_r, \Sigma_r$ là trung bình và ma trận hiệp phương sai của đặc trưng từ hình ảnh thực.
  - $\mu_g, \Sigma_g$ là trung bình và ma trận hiệp phương sai của đặc trưng từ hình ảnh sinh ra.
  - $\text{Tr}$ là phép toán trace, tính tổng các phần tử trên đường chéo của ma trận.

**Áp dụng**: 
- FID là chỉ số phổ biến để đánh giá các mô hình sinh hình ảnh, đặc biệt là trong các bài toán như **image generation** hoặc **style transfer**.
- **Ưu điểm**: FID cung cấp một chỉ số chính xác và dễ hiểu về sự khác biệt giữa các phân phối hình ảnh thực và sinh ra.
- **Nhược điểm**: FID phụ thuộc vào mô hình phân loại Inception-v3, do đó có thể không phù hợp cho tất cả các loại dữ liệu.

---

### **3. Perceptual Path Length (PPL)**

**Định nghĩa**: PPL là một chỉ số dùng để đo lường sự mượt mà và liên tục trong không gian hình ảnh sinh ra của các mô hình GAN. Nó giúp xác định xem mô hình có thể sinh ra những biến thể ảnh khác nhau của cùng một đối tượng mà không gây ra những sự thay đổi lớn về hình ảnh không cần thiết.

- **Công thức**: 
$$
  PPL = \frac{1}{N} \sum_{i=1}^{N} \| \mathbf{z}_i - \mathbf{z}_{i+1} \|
$$
  Trong đó:
  - $\mathbf{z}_i$ và $\mathbf{z}_{i+1}$ là các vector trong không gian latent của mô hình GAN.
  - N là số lượng mẫu trong không gian latent.

**Áp dụng**: 
- Sử dụng trong các mô hình GAN để đánh giá độ mượt mà trong quá trình sinh ảnh và sự liên tục của không gian hình ảnh.
- **Ưu điểm**: Đo lường độ mượt mà của quá trình sinh và giúp tránh việc sinh ra các hình ảnh không tự nhiên.

---

### **4. Log-Likelihood**

**Định nghĩa**: Log-Likelihood là một chỉ số dùng để đo lường mức độ chính xác mà mô hình sinh (ví dụ như VAE hoặc các mô hình sinh xác suất) có thể tái tạo được các mẫu từ phân phối thực tế. Đặc biệt, trong các mô hình sinh xác suất, Log-Likelihood giúp xác định mức độ phù hợp giữa phân phối sinh ra và phân phối thực.

- **Công thức**:
$$
  \mathcal{L}(\theta) = \sum_{i=1}^{N} \log p(x_i | \theta)
$$
  Trong đó:
  - $x_i$ là mẫu thực tế.
  - $p(x_i | \theta)$ là xác suất mô hình sinh ra mẫu $x_i$.
  - $\theta$ là các tham số của mô hình sinh.

**Áp dụng**: 
- Được sử dụng trong các mô hình sinh xác suất như Variational Autoencoders (VAE).
- **Ưu điểm**: Giúp đánh giá mức độ khớp của mô hình sinh với dữ liệu thực tế.

---

### **5. Human Evaluation**

**Định nghĩa**: Đánh giá chất lượng sinh thông qua ý kiến chủ quan của con người. Đây là một phương pháp quan trọng, đặc biệt trong các mô hình sinh văn bản (như GPT-3), sinh ảnh, hoặc sinh nhạc.

- **Áp dụng**: Được sử dụng khi không thể đo lường chất lượng của mô hình chỉ bằng các chỉ số số học (đặc biệt trong các tác vụ sáng tạo như sinh văn bản, âm nhạc, hoặc hình ảnh nghệ thuật).
- **Ví dụ**: Đánh giá văn bản sinh ra bởi GPT-3 về tính sáng tạo, mạch lạc và tự nhiên.
- **Phương pháp**: Người dùng sẽ được yêu cầu đánh giá văn bản hoặc hình ảnh sinh ra dựa trên các tiêu chí như tính sáng tạo, sự mạch lạc, tính tự nhiên, v.v.

---

### **6. Diversity Metrics**

**Định nghĩa**: Đánh giá độ đa dạng của các mẫu sinh ra từ mô hình. Những mô hình sinh (đặc biệt là trong bài toán tạo hình ảnh, âm thanh hoặc văn bản) có thể tạo ra nhiều mẫu giống nhau hoặc thiếu tính đa dạng.

- **Công thức**: Các chỉ số như **Intra-class diversity** và **Inter-class diversity** được sử dụng để đo lường độ đa dạng này. Ví dụ:
$$
  \text{Intra-class diversity} = \frac{1}{M} \sum_{i=1}^{M} \text{distance}(x_i, x_j) \quad \text{for } i \neq j
$$
  Trong đó $x_i$ là các mẫu được sinh ra và distance là khoảng cách giữa chúng (ví dụ, Euclidean distance).

**Áp dụng**: 
- Được sử dụng để đảm bảo rằng mô hình sinh ra các mẫu đa dạng thay vì chỉ tập trung vào một số kiểu mẫu nhất định.

---

### **Tóm Tắt**

Các **evaluation metrics** cho mô hình sinh (Generative AI) giúp đánh giá chất lượng và hiệu quả của các mô hình sinh như GANs, VAEs, và các mô hình sinh văn bản như GPT. Các chỉ số như **Inception Score (IS)**, **Fréchet Inception Distance (FID)**, **Perceptual Path Length (PPL)**, **Log-Likelihood**, và **Human Evaluation** là những chỉ số quan trọng và được sử dụng rộng rãi để đảm bảo rằng mô hình sinh không chỉ tạo ra các mẫu chất lượng cao mà còn có sự đa dạng và tính tự nhiên trong các tác phẩm sáng tạo.



## 3. Other Evaluation Metrics
Ngoài các **evaluation metrics** đã đề cập cho các mô hình sinh (Generative AI), trong các bài toán và nghiên cứu về **AI** và **Machine Learning** nói chung, còn có rất nhiều loại **metrics** khác được sử dụng tùy theo loại bài toán, mô hình và mục đích nghiên cứu. Các loại **evaluation metrics** này có thể được chia thành các nhóm lớn dựa trên kiểu bài toán, bao gồm:

### 1. **Evaluation Metrics cho Bài Toán Học Máy Cổ Điển**
#### **1.1. Phân Loại (Classification Metrics)**

- **Accuracy (Độ Chính Xác)**: 
  Đo lường tỷ lệ các dự đoán đúng trong tổng số dự đoán. 
$$
  \text{Accuracy} = \frac{\text{Số lượng dự đoán đúng}}{\text{Tổng số dự đoán}}
$$
  - **Ứng dụng**: Sử dụng cho các bài toán phân loại nhị phân và đa lớp.

- **Precision (Độ Chính Xác)**:
  Đo lường tỉ lệ giữa các dự đoán chính xác (true positive) trên tổng số dự đoán là dương tính (true positive + false positive).
$$
  \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$
  - **Ứng dụng**: Được sử dụng trong các trường hợp mà các lỗi **false positive** là nghiêm trọng, ví dụ như trong phát hiện gian lận.

- **Recall (Độ Nhạy)**:
  Đo lường tỉ lệ giữa các dự đoán chính xác (true positive) trên tổng số trường hợp thực sự là dương tính (true positive + false negative).
$$
  \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$
  - **Ứng dụng**: Thích hợp khi các lỗi **false negative** nghiêm trọng, ví dụ như trong phát hiện bệnh.

- **F1-Score**:
  Là trung bình điều hòa của Precision và Recall, giúp cân bằng giữa chúng.
$$
  F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$
  - **Ứng dụng**: Được sử dụng khi có sự cần thiết phải cân bằng giữa Precision và Recall, đặc biệt trong các bài toán bất cân xứng dữ liệu.

- **AUC-ROC (Area Under the Curve - Receiver Operating Characteristic)**:
  AUC là diện tích dưới đường cong ROC, biểu diễn khả năng phân biệt giữa các lớp trong mô hình.
$$
  \text{AUC} = \int_{0}^{1} \text{True Positive Rate} \, d(\text{False Positive Rate})
$$
  - **Ứng dụng**: Được sử dụng để đánh giá mô hình phân loại trong trường hợp dữ liệu không cân bằng.

- **Confusion Matrix (Ma Trận Nhầm Lẫn)**:
  Là bảng tóm tắt các dự đoán phân loại so với giá trị thực tế, giúp nhìn thấy rõ hơn các loại lỗi trong mô hình.

#### **1.2. Phân Tích Khả Năng Dự Đoán với Dự Báo (Regression Metrics)**

- **Mean Absolute Error (MAE)**:
  Là trung bình của các giá trị tuyệt đối giữa giá trị dự đoán và giá trị thực tế.
$$
  \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y_i}|
$$
  - **Ứng dụng**: Sử dụng trong bài toán dự báo liên tục khi cần tính toán sai số trung bình tuyệt đối.

- **Mean Squared Error (MSE)**:
  Là trung bình của bình phương sai số giữa giá trị thực tế và giá trị dự đoán.
$$
  \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2
$$
  - **Ứng dụng**: Được sử dụng rộng rãi trong các bài toán hồi quy, nhưng có thể bị ảnh hưởng mạnh mẽ bởi các giá trị ngoại lai (outliers).

- **Root Mean Squared Error (RMSE)**:
  Là căn bậc hai của MSE, giúp đưa kết quả về cùng đơn vị với dữ liệu gốc.
$$
  \text{RMSE} = \sqrt{\text{MSE}}
$$
  - **Ứng dụng**: Thích hợp khi bạn cần biểu diễn sai số trong cùng một đơn vị với dữ liệu gốc.

- **R-squared (R²)**:
  Đo lường mức độ phù hợp của mô hình với dữ liệu.
$$
  R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y_i})^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$
  - **Ứng dụng**: Được sử dụng để đánh giá mức độ của mô hình hồi quy.

---

### 2. **Evaluation Metrics cho Mô Hình Sinh (Generative Models)**

Như đã thảo luận trước đó, các mô hình sinh (Generative AI) yêu cầu một số **evaluation metrics** đặc thù như:

- **Inception Score (IS)**
- **Fréchet Inception Distance (FID)**
- **Perceptual Path Length (PPL)**
- **Log-Likelihood**

---

### 3. **Evaluation Metrics cho Mô Hình Học Sâu (Deep Learning Models)**

- **Loss Function**:
  Loss function là một chỉ số quan trọng trong việc huấn luyện các mô hình học sâu. Tùy thuộc vào loại bài toán mà ta sử dụng loss function khác nhau.
  - **Cross-Entropy Loss** (cho phân loại): 
  $$
    L = - \sum_{i=1}^{n} y_i \log(\hat{y}_i)
  $$
  - **Mean Squared Error Loss** (cho hồi quy): 
  $$
    L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  $$

- **BLEU Score** (đánh giá chất lượng dịch máy):
  Là một metric phổ biến trong các bài toán dịch máy (machine translation), dựa trên sự tương đồng giữa văn bản sinh ra và văn bản tham chiếu.
$$
  BLEU = BP \times \exp\left( \sum_{n=1}^{N} w_n \log p_n \right)
$$
  Trong đó \(p_n\) là tỷ lệ n-gram chính xác, và \(BP\) là penalty cho việc giảm độ dài của câu.

- **Mean Average Precision (MAP)**: 
  Được sử dụng trong các bài toán **information retrieval** (truy xuất thông tin), để đánh giá mức độ chính xác của hệ thống khi trả về các kết quả theo thứ tự.

---

### 4. **Evaluation Metrics cho Mô Hình Reinforcement Learning**

- **Cumulative Reward**:
  Đo lường tổng phần thưởng mà một tác nhân (agent) nhận được trong suốt quá trình tương tác với môi trường.
$$
  \text{Cumulative Reward} = \sum_{t=1}^{T} r_t
$$
  Trong đó \(r_t\) là phần thưởng tại thời điểm \(t\).

- **Average Return**:
  Đo lường giá trị trung bình của phần thưởng trong một số tập thử nghiệm hoặc episodes.
$$
  \text{Average Return} = \frac{1}{N} \sum_{i=1}^{N} R_i
$$
  Trong đó \(R_i\) là tổng phần thưởng từ tập thử nghiệm thứ \(i\).

- **Success Rate**:
  Đo lường tỷ lệ thành công trong một tập thử nghiệm, ví dụ như tỷ lệ hoàn thành mục tiêu trong môi trường học tập.

---

### 5. **Evaluation Metrics cho Mô Hình Thị Giác Máy Tính (Computer Vision)**

- **Intersection over Union (IoU)**:
  Được sử dụng trong các bài toán phân đoạn ảnh (image segmentation) để đánh giá độ chính xác của phân đoạn giữa hai vùng.
$$
  IoU = \frac{|A \cap B|}{|A \cup B|}
$$
  Trong đó \(A\) và \(B\) là các tập hợp vùng phân đoạn.

- **Pixel Accuracy**:
  Tính tỷ lệ số pixel đúng dự đoán trong phân đoạn ảnh so với tổng số pixel.
$$
  \text{Pixel Accuracy} = \frac{\text{Số pixel đúng}}{\text{Tổng số pixel}}
$$

---

### Kết luận

Các **evaluation metrics** là yếu tố quan trọng trong việc đánh giá và so sánh các mô hình **AI/ML**. Chúng giúp xác định mức độ hiệu quả của mô hình đối với bài toán cụ thể, cũng như giúp các nhà nghiên cứu và kỹ sư điều chỉnh và tối ưu mô hình. Việc chọn lựa metric phù hợp với bài toán sẽ giúp cải thiện hiệu quả của các mô hình trong thực tế.