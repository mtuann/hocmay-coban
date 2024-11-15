### Mô Hình Diffusion: Chi Tiết, Input, Output, Cách Huấn Luyện, Công Thức Toán Học và Các Biến Thể

Mô hình **Diffusion** là một loại mô hình học máy mạnh mẽ được sử dụng trong các tác vụ sinh ảnh (image generation) và tạo ra dữ liệu mới dựa trên một quá trình ngược lại (reverse process). Diffusion đã nổi lên như một trong những phương pháp chính trong việc tạo ra hình ảnh từ nhiễu (noise), và nó được xem là một đối thủ mạnh của các mô hình GAN (Generative Adversarial Networks) trong một số trường hợp.

Mô hình Diffusion có thể được sử dụng trong nhiều ứng dụng như sinh ảnh, xử lý ảnh, tăng cường dữ liệu, và nhiều tác vụ khác. Dưới đây là thông tin chi tiết về mô hình Diffusion, các công thức toán học liên quan, cách huấn luyện, ví dụ, và các biến thể của nó.

### 1. **Giới Thiệu về Diffusion Models**

Mô hình Diffusion là một loại mô hình generative được xây dựng dựa trên ý tưởng **reverse diffusion process** (quá trình khuếch tán ngược). Quá trình khuếch tán này mô phỏng một chuỗi bước liên tiếp, trong đó dữ liệu gốc (như ảnh, văn bản) bị "nhiễu hóa" qua các bước để trở thành một phân phối nhiễu. Sau đó, mô hình học cách "hồi phục" (reconstruct) dữ liệu từ nhiễu này qua quá trình khuếch tán ngược.

#### **Quá Trình Khuếch Tán (Forward Diffusion Process)**

- **Mục tiêu** của quá trình khuếch tán là chuyển đổi một dữ liệu gốc $x_0$ (ví dụ, hình ảnh) thành một nhiễu (noise) $x_T$ thông qua một chuỗi các bước nhiễu hóa (diffusion steps).
  
  Quá trình này được mô phỏng bởi một chuỗi $x_0, x_1, ..., x_T$, trong đó mỗi bước $x_t$ là dữ liệu sau khi nhiễu hóa ở bước $t$.
  
- **Forward Process** (quá trình khuếch tán): Quá trình khuếch tán là một chuỗi các bước dần dần thêm nhiễu vào dữ liệu gốc. Mỗi bước có thể được mô phỏng bằng một hàm nhiễu $\epsilon$.

Công thức cho quá trình khuếch tán là:

$$
x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon_t
$$

Trong đó:
- $x_t$ là dữ liệu sau $t$-th step (nhiễu hóa).
- $\epsilon_t$ là nhiễu ngẫu nhiên (thường được lấy từ phân phối chuẩn $\mathcal{N}(0, I)$).
- $\alpha_t$ là một hệ số điều chỉnh mức độ nhiễu tại mỗi bước.

### 2. **Quá Trình Khuếch Tán Ngược (Reverse Diffusion Process)**

- **Reverse Diffusion** là quá trình học cách đảo ngược quá trình khuếch tán: từ nhiễu $x_T$ trở về dữ liệu gốc $x_0$. Mô hình học cách tái tạo dữ liệu $x_0$ từ nhiễu.

Quá trình ngược (reverse process) có thể được mô hình hóa bằng cách sử dụng một hàm $p_\theta(x_{t-1} | x_t)$ dựa trên mạng nơ-ron. Mô hình học cách dự đoán $x_{t-1}$ từ $x_t$, qua các bước ngược.

Công thức cho quá trình ngược (reverse diffusion) là:

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_\theta^2(t))
$$

Trong đó:
- $x_{t-1}$ là dữ liệu dự đoán từ $x_t$.
- $\mu_\theta(x_t, t)$ và $\sigma_\theta^2(t)$ là các tham số học được, xác định mức độ nhiễu và cách dự đoán dữ liệu từ $x_t$.
- Hàm này được học từ dữ liệu huấn luyện.

### 3. **Công Thức Toán Học và Các Biến Thể**

#### **Loss Function**

Để huấn luyện mô hình Diffusion, ta sử dụng một **loss function** đo lường sự khác biệt giữa đầu ra của mô hình (dự đoán $x_0$ từ $x_T$) và dữ liệu gốc. Loss function thường được định nghĩa như sau:

$$
\mathcal{L}(\theta) = \mathbb{E}_{q(x_0)} \left[ D_{KL} \left( q(x_T | x_0) \| p_\theta(x_T) \right) \right]
$$

$$
\mathcal{L}(\theta) = \mathbb{E}_{q(x_0)} \left[ D_{\text{KL}} \left( q(x_T | x_0) \| p_\theta(x_T) \right) \right]
$$


Trong đó:
- $q(x_T | x_0)$ là phân phối dữ liệu theo quá trình khuếch tán (forward process).
- $p_\theta(x_T)$ là phân phối dữ liệu theo mô hình ngược (reverse process).
- $D_{KL}$ là độ đo phân phối Kullback-Leibler divergence, đo lường sự khác biệt giữa các phân phối.

Một số mô hình Diffusion gần đây đã cải tiến loss function này, sử dụng **denoising score matching** để cải thiện khả năng tái tạo hình ảnh.

#### **Biến Thể của Mô Hình Diffusion**

1. **DDPM (Denoising Diffusion Probabilistic Models)**:
   - Đây là một trong những biến thể đầu tiên của mô hình Diffusion. DDPM áp dụng một quá trình khuếch tán dần dần và sử dụng phương pháp **denoising score matching** trong quá trình huấn luyện.

2. **Score-based Models**:
   - Các mô hình này học một **score function** để đo lường độ dốc của log-likelihood, từ đó phục hồi dữ liệu gốc từ nhiễu. Công thức score-based là:

   $$
   \nabla_x \log p(x) = \text{score function}
   $$

3. **LDM (Latent Diffusion Models)**:
   - Là một biến thể sử dụng không gian **latent** để giảm thiểu chi phí tính toán. Các mô hình này áp dụng quá trình khuếch tán trong không gian tiềm ẩn thay vì không gian ảnh trực tiếp, giúp cải thiện hiệu suất và giảm độ phức tạp.

4. **Guided Diffusion Models**:
   - Mô hình này sử dụng **conditioning signal** (thường là văn bản hoặc các thông tin khác) để hướng dẫn quá trình khuếch tán, cho phép tạo ra các kết quả có hướng dẫn (ví dụ: tạo ảnh từ mô tả văn bản).

5. **Improved Denoising Diffusion Models**:
   - Các biến thể cải tiến của DDPM, chẳng hạn như **Improved DDPM** (iDDPM), nhằm tăng tốc quá trình huấn luyện và cải thiện chất lượng kết quả đầu ra.

### 4. **Cách Huấn Luyện (Training) Mô Hình Diffusion**

#### **Training Pipeline**:

1. **Tiền Xử Lý Dữ Liệu**:
   - Chuẩn bị dữ liệu huấn luyện (ví dụ, hình ảnh) và chuyển nó thành các token hoặc các vector phù hợp để đưa vào mô hình.
   
2. **Xây Dựng Quá Trình Khuếch Tán**:
   - Áp dụng quá trình khuếch tán (forward diffusion) lên dữ liệu đầu vào để tạo ra nhiễu.
   
3. **Huấn Luyện Mô Hình**:
   - Sử dụng loss function để tối ưu các tham số của mô hình. Mô hình học cách dự đoán các bước ngược từ nhiễu, từ đó khôi phục lại dữ liệu gốc.
   
4. **Generative Process**:
   - Sau khi huấn luyện, mô hình có thể được sử dụng để sinh ra dữ liệu mới. Quá trình này bắt đầu từ nhiễu và sử dụng các bước ngược để tái tạo dữ liệu.

### 5. **Ví Dụ Cụ Thể**

Giả sử bạn muốn sinh ra một hình ảnh từ mô tả văn bản sử dụng mô hình Diffusion:

1. **Input**: "Một con mèo đang ngồi trên bãi cỏ".
2. Mô hình Diffusion sẽ thực hiện quá trình:
   - Lấy một nhiễu ngẫu nhiên (Gaussian noise).
   - Áp dụng quá trình ngược để tạo ra hình ảnh từ nhiễu, theo cách mô phỏng dữ liệu ảnh gốc.

3. **Output**: Một hình ảnh có thể là: "Một con mèo thật đang ngồi trên cỏ".

### 6. **Tóm Tắt**

Mô hình Diffusion là một công cụ mạnh mẽ trong việc tạo ra dữ liệu mới từ nhiễu, sử dụng quá trình khuếch tán ngược để tái tạo dữ liệu từ nhiễu. Các bước

 chính trong mô hình Diffusion bao gồm:
- Quá trình khuếch tán (forward diffusion).
- Quá trình khuếch tán ngược (reverse diffusion).
- Loss function dựa trên Kullback-Leibler divergence và denoising score matching.
- Các biến thể như DDPM, Score-based models, Latent Diffusion, và Guided Diffusion.

Mô hình Diffusion có thể áp dụng trong nhiều ứng dụng, bao gồm tạo ảnh, chuyển đổi kiểu ảnh, và các tác vụ sinh dữ liệu khác.