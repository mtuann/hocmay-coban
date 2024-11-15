### **Generative Models: Tổng Quan, Công Thức Toán Học, Ví Dụ và Input/Output**

**Generative Models** (Mô Hình Sinh) là các mô hình học máy có khả năng học từ dữ liệu và tạo ra dữ liệu mới tương tự với dữ liệu huấn luyện. Các mô hình này không chỉ dự đoán nhãn (như trong mô hình phân loại) mà còn có thể tạo ra các mẫu (samples) mới từ phân phối dữ liệu mà chúng đã học.

Các **Generative Models** có thể được chia thành nhiều loại khác nhau, chẳng hạn như:

- **Generative Adversarial Networks (GANs)**
- **Variational Autoencoders (VAEs)**
- **Autoregressive Models (Ví dụ: PixelCNN, WaveNet)**

Dưới đây là thông tin chi tiết về các mô hình generative, công thức toán học, và các ví dụ cụ thể.

---

### **1. Mô Hình GAN (Generative Adversarial Networks)**

**Generative Adversarial Networks (GANs)** là một trong những mô hình generative nổi bật nhất. GANs bao gồm hai mạng neuron:

- **Generator**: Mạng này tạo ra dữ liệu giả từ một phân phối xác suất đơn giản (ví dụ: phân phối chuẩn).
- **Discriminator**: Mạng này cố gắng phân biệt giữa dữ liệu thật (từ bộ dữ liệu huấn luyện) và dữ liệu giả (do generator tạo ra).

#### **Công thức toán học trong GAN**

GANs dựa trên lý thuyết trò chơi, nơi generator và discriminator đối kháng nhau.

- **Objective của Generator**: Generator cố gắng làm cho dữ liệu giả mà nó tạo ra ngày càng giống với dữ liệu thật. Mục tiêu là làm cho **Discriminator** không thể phân biệt giữa dữ liệu thật và dữ liệu giả.
  
- **Objective của Discriminator**: Discriminator cố gắng phân biệt giữa dữ liệu thật và dữ liệu giả. Nếu discriminator phân biệt chính xác, thì đó là dấu hiệu của việc huấn luyện tốt.

Mục tiêu của GAN là tối ưu hóa hàm mất mát tổng hợp giữa Generator và Discriminator. Hàm mất mát của GAN có thể được mô tả như sau:

$$
\mathcal{L}_{\text{GAN}} = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$

Trong đó:
- $x$ là dữ liệu thật từ phân phối $p_{\text{data}}(x)$,
- $z$ là vector ngẫu nhiên từ phân phối $p_z(z)$,
- $G(z)$ là dữ liệu giả được tạo ra bởi Generator,
- $D(x)$ là xác suất mà Discriminator cho rằng $x$ là thật.

#### **Input và Output trong GAN**
- **Input**: Một vector ngẫu nhiên $z$ (thường là một vector với phân phối chuẩn) được cung cấp cho Generator.
- **Output**: Dữ liệu giả $G(z)$, có thể là hình ảnh, âm thanh, văn bản, v.v.
  
**Ví dụ**:
- Trong việc tạo hình ảnh, $z$ có thể là một vector ngẫu nhiên, và $G(z)$ sẽ là một bức ảnh được tạo ra. Để huấn luyện mô hình, Discriminator sẽ cố gắng phân biệt giữa hình ảnh thật từ bộ dữ liệu huấn luyện và hình ảnh giả từ Generator.

---

### **2. Variational Autoencoders (VAEs)**

**Variational Autoencoders (VAEs)** là một mô hình generative dựa trên cấu trúc mạng autoencoder, nhưng có sự mở rộng để mô phỏng phân phối xác suất của dữ liệu. VAEs giúp mô hình hóa dữ liệu theo dạng phân phối tiềm ẩn, thay vì chỉ ánh xạ từ không gian dữ liệu sang không gian tiềm ẩn.

#### **Công thức toán học trong VAEs**

VAE sử dụng phương pháp **bayesian inference** để học phân phối xác suất của dữ liệu.

- **Encoder**: Mạng encoder $q(z|x)$ ánh xạ đầu vào $x$ thành một phân phối tiềm ẩn $z$, mô phỏng phân phối tiềm ẩn.
- **Decoder**: Mạng decoder $p(x|z)$ tái tạo dữ liệu $x$ từ phân phối tiềm ẩn $z$.

Hàm mất mát trong VAE bao gồm hai phần:
1. **Reconstruction loss**: Phần này đo lường sự khác biệt giữa dữ liệu đầu vào $x$ và dữ liệu tái tạo từ decoder $\hat{x}$.
2. **KL divergence loss**: Phần này điều chỉnh phân phối tiềm ẩn sao cho gần với phân phối chuẩn $p(z)$ (thường là phân phối chuẩn đơn giản).

Hàm mất mát của VAE có thể được viết như sau:

$$
\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{\text{KL}}[q(z|x) \parallel p(z)]
$$

Trong đó:
- $\mathbb{E}_{q(z|x)}[\log p(x|z)]$ là khả năng tái tạo dữ liệu từ phân phối tiềm ẩn,
- $D_{\text{KL}}[q(z|x) \parallel p(z)]$ là độ lệch KL giữa phân phối tiềm ẩn $q(z|x)$ và phân phối chuẩn $p(z)$.

#### **Input và Output trong VAE**
- **Input**: Một dữ liệu đầu vào $x$, ví dụ là một hình ảnh.
- **Output**: Một dữ liệu tái tạo $\hat{x}$, được tạo ra từ phân phối tiềm ẩn $z$.

**Ví dụ**:
- Nếu đầu vào là một hình ảnh, VAE sẽ mã hóa hình ảnh này vào một không gian tiềm ẩn $z$, sau đó tái tạo lại hình ảnh từ không gian tiềm ẩn đó.

---

### **3. Autoregressive Models (Ví Dụ: PixelCNN, WaveNet)**

Các **autoregressive models** là những mô hình generative tạo ra dữ liệu một cách tuần tự, nơi mỗi giá trị được tạo ra dựa trên các giá trị đã tạo ra trước đó. Các mô hình này đặc biệt hữu ích cho các loại dữ liệu có tính tuần tự, như hình ảnh hoặc âm thanh.

#### **Công thức toán học trong Autoregressive Models**

Mô hình autoregressive có thể được biểu diễn bằng công thức sau, nơi $x_1, x_2, \dots, x_T$ là các giá trị của dữ liệu đầu ra (ví dụ, các pixel của hình ảnh hoặc các mẫu âm thanh):

$$
P(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} P(x_t | x_1, x_2, \dots, x_{t-1})
$$

Mỗi giá trị $x_t$ được dự đoán dựa trên các giá trị trước đó. Ví dụ, trong **PixelCNN**, mỗi pixel trong hình ảnh được dự đoán dựa trên các pixel trước đó.

#### **Input và Output trong Autoregressive Models**
- **Input**: Dữ liệu được tạo ra từng bước, ví dụ là các pixel của hình ảnh hoặc các mẫu âm thanh.
- **Output**: Dữ liệu được tạo ra liên tục, ví dụ là hình ảnh hoặc âm thanh mới.

**Ví dụ**:
- Trong **PixelCNN**, mô hình sẽ tạo ra từng pixel của hình ảnh một cách tuần tự, dựa trên các pixel đã tạo ra trước đó.

---

### **Tóm Tắt về Generative Models**

- **Generative Models** giúp tạo ra dữ liệu mới từ phân phối xác suất đã học từ dữ liệu huấn luyện.
- Các ví dụ bao gồm **GANs**, **VAEs**, và các **Autoregressive Models** như **PixelCNN**.
- Các công thức toán học trong generative models thường liên quan đến tối ưu hóa hàm mất mát dựa trên phân phối xác suất (ví dụ: hàm mất mát trong GANs hoặc VAE).
- Các **input** và **output** trong generative models có thể là bất kỳ loại dữ liệu nào (hình ảnh, văn bản, âm thanh), và quá trình học cho phép mô hình tạo ra dữ liệu mới tương tự như dữ liệu huấn luyện.

--- 

Các mô hình generative hiện nay đã có những ứng dụng thực tế rộng rãi, bao gồm tạo ảnh (như GANs), tạo văn bản (như GPT), và tạo âm thanh (như WaveNet).