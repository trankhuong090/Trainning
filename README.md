# 🏛️ HOSE QUANT PORTFOLIO | SMA + OBV (OR) & MPT OPTIMIZATION

Ứng dụng Web App tương tác xây dựng trên nền tảng **Streamlit** nhằm kiểm định tính hiệu quả của chiến lược giao dịch kỹ thuật kết hợp tín hiệu **SMA (Simple Moving Average)** và **OBV (On-Balance Volume)** theo logic **OR**, kết hợp với mô hình phân bổ danh mục đầu tư theo lý thuyết danh mục hiện đại **Modern Portfolio Theory (MPT - Markowitz)** so với **Equal Weight (1/N)** trên dữ liệu lịch sử thị trường chứng khoán Việt Nam (HOSE 2020 – 2022).

---

## 📌 1. Bối Cảnh & Mục Tiêu Nghiên Cứu

Đề tài thuộc môn học **Quản Trị Danh Mục Đầu Tư (Chương trình Cao học)** với các mục tiêu trọng tâm:
1. **Sàng lọc cổ phiếu định lượng:** Xây dựng mô hình chấm điểm tổ hợp trên dữ liệu huấn luyện (Train 2020–2021) dựa trên 4 tiêu chí: *Annual Return (30%)*, *Sharpe Ratio (30%)*, *Liquidity (25%)*, và *Max Drawdown (15%)*.
2. **Đa dạng hóa danh mục ngành:** Lựa chọn 5 mã cổ phiếu tiêu biểu đại diện cho 5 nhóm ngành kinh tế then chốt:
   - **DIG** — Bất Động Sản (Nhóm chu kỳ, beta cao)
   - **DGC** — Hóa Chất & Phốt Pho (Cổ phiếu tăng trưởng cơ bản xuất sắc)
   - **VND** — Chứng Khoán & Dịch Vụ Tài Chính (Thanh khoản bùng nổ)
   - **HAH** — Vận Tải Biển / Logistics (Chuỗi cung ứng toàn cầu)
   - **MSN** — Hàng Tiêu Dùng Thiết Yếu (Phòng thủ, cân bằng danh mục)
3. **Chiến lược giao dịch SMA + OBV (OR logic):** Tận dụng tính xu hướng của đường giá (SMA Crossover) kết hợp dòng tiền thông minh (OBV Crossover) để ra quyết định Mua (Buy) / Bán (Sell) kịp thời.
4. **Nguyên tắc chống Look-Ahead Bias:** Toàn bộ tín hiệu phát sinh tại phiên $t$ chỉ được thực thi khớp lệnh từ phiên $t+1$ (`shift(1)`), đảm bảo tính phản ánh chân thực điều kiện giao dịch thực tế trên sàn HOSE.
5. **Kiểm định ngoài mẫu (Out-of-Sample Test 2022):** Đánh giá năng lực bảo toàn vốn và quản trị rủi ro khi thị trường bước vào pha giảm điểm sâu (Bear Market 2022).

---

## 🧠 2. Phương Pháp Luận & Cơ Sở Toán Học

### 2.1. Tín Hiệu Kỹ Thuật Kết Hợp (OR Logic)
- **Tín hiệu SMA:**
  - $\text{Buy}_{\text{SMA}} = 1 \iff \text{SMA}_{\text{short}} > \text{SMA}_{\text{long}} \land \text{SMA}_{\text{short}}(t-1) \le \text{SMA}_{\text{long}}(t-1)$
  - $\text{Sell}_{\text{SMA}} = -1 \iff \text{SMA}_{\text{short}} < \text{SMA}_{\text{long}} \land \text{SMA}_{\text{short}}(t-1) \ge \text{SMA}_{\text{long}}(t-1)$
- **Tín hiệu OBV:**
  - $\text{Buy}_{\text{OBV}} = 1 \iff \text{OBV} > \text{MA}(\text{OBV}) \land \text{OBV}(t-1) \le \text{MA}(\text{OBV})(t-1)$
  - $\text{Sell}_{\text{OBV}} = -1 \iff \text{OBV} < \text{MA}(\text{OBV}) \land \text{OBV}(t-1) \ge \text{MA}(\text{OBV})(t-1)$
- **Quy tắc kết hợp OR:**
  $$\text{Tín hiệu Mua} \iff \text{SMA Mua} \lor \text{OBV Mua}$$
  $$\text{Tín hiệu Bán} \iff \text{SMA Bán} \lor \text{OBV Bán}$$
  *(Nếu xảy ra xung đột cùng phiên: Tín hiệu = 0, giữ trạng thái trung lập)*

### 2.2. Phân Bổ Danh Mục: MPT vs Equal Weight
- **Danh mục Equal Weight (1/N):**
  $$w_i = \frac{1}{N} = \frac{1}{5} = 20\%$$
- **Danh mục MPT (Markowitz Max Sharpe trên Train):**
  $$\max_{\mathbf{w}} \text{Sharpe}(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{\mu} - R_f}{\sqrt{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}}} \quad \text{với ràng buộc:} \quad \sum_{i=1}^N w_i = 1, \quad 0 \le w_i \le 1$$

---

## 📊 3. Bảng Kết Quả Thực Nghiệm Ngoài Mẫu (Test 2022)

Trong năm 2022, VN-Index sụt giảm mạnh mẽ khiến chiến lược thụ động Buy & Hold chịu tổn thất nặng nề:

| Tiêu Chí Hiệu Suất | 1. Equal Weight Buy & Hold | 2. SMA+OBV (Equal Weight) | 3. SMA+OBV (MPT Markowitz) |
| :--- | :---: | :---: | :---: |
| **Tổng Lợi Nhuận (Total Return)** | **-64.53%** | **-16.57%** | **-14.84%** |
| **Lợi Nhuận Năm Hóa (CAGR)** | -64.92% | -16.75% | -15.00% |
| **Sharpe Ratio** | -1.54 | -0.94 | **-0.78** |
| **Sụt Giảm Tối Đa (Max Drawdown)** | -65.34% | -28.90% | **-26.68%** |
| **Chênh Lệch Alpha so với B&H** | *Benchmark* | **+47.96%** | **+49.69%** |

> **Nhận xét cốt lõi:** Chiến lược chủ động kết hợp SMA + OBV (OR) phát huy năng lực bảo vệ tài sản vượt bậc. Trong khi danh mục Buy & Hold mất gần 2/3 tổng tài sản, hệ thống đã chủ động hạ vị thế về tiền mặt (Cash), cắt lỗ thành công và giữ lại phần lớn giá trị danh mục. Mô hình MPT tối ưu phương sai - hiệp phương sai trên tập Train tiếp tục duy trì ưu thế ngoài mẫu so với Equal Weight.

---

## 🚀 4. Cài Đặt & Chạy Ứng Dụng Cục Bộ (Local Run)

### Yêu Cầu Môi Trường
- Python 3.10, 3.11 hoặc 3.12
- Git (để quản lý phiên bản)

### Các Bước Thực Hiện
1. **Clone repository:**
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Tạo và kích hoạt môi trường ảo (khuyến nghị):**
   - Trên Windows:
     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - Trên macOS / Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Cài đặt các gói thư viện cần thiết:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Khởi chạy ứng dụng Streamlit:**
   ```bash
   streamlit run app.py
   ```
   Ứng dụng sẽ tự động mở tại địa chỉ `http://localhost:8501` trên trình duyệt.

---

## 🌐 5. Hướng Dẫn Deploy Lên Streamlit Community Cloud (Miễn Phí)

Deploy ứng dụng lên web trong vòng 3 phút để chia sẻ với giảng viên, hội đồng phản biện hoặc đồng nghiệp:

1. **Đưa mã nguồn lên GitHub:**
   - Tạo một Repository mới trên [GitHub.com](https://github.com) (ví dụ đặt tên: `hose-quant-backtest`).
   - Đẩy 4 file lên GitHub:
     - `app.py` (Mã nguồn ứng dụng)
     - `requirements.txt` (Danh sách thư viện phụ thuộc)
     - `README.md` (Tài liệu hướng dẫn & giải thích)
     - `HOSE_2020_2023_in.csv` (File dữ liệu 100 cổ phiếu HOSE)
   - Lệnh git cơ bản:
     ```bash
     git init
     git add .
     git commit -m "Initial commit for HOSE Quant App"
     git branch -M main
     git remote add origin https://github.com/<your-username>/hose-quant-backtest.git
     git push -u origin main
     ```

2. **Kết nối và Deploy trên Streamlit Cloud:**
   - Truy cập [share.streamlit.io](https://share.streamlit.io) và đăng nhập bằng tài khoản GitHub.
   - Nhấn nút **"New app"** (hoặc **"Create app"**).
   - Điền thông tin ứng dụng:
     - **Repository:** `<your-username>/hose-quant-backtest`
     - **Branch:** `main`
     - **Main file path:** `app.py`
   - Nhấn **"Deploy!"**.
   - Chờ hệ thống tự động build container trong 1–2 phút. Ứng dụng web công khai sẽ sẵn sàng và cung cấp đường link truy cập trực tuyến (ví dụ: `https://hose-quant-backtest.streamlit.app`).

---

## 📂 6. Cấu Trúc Thư Mục Repository

```text
├── app.py                      # Mã nguồn chính của ứng dụng Streamlit Dashboard
├── requirements.txt            # Danh sách thư viện Python cần thiết
├── README.md                   # Tài liệu học thuật & hướng dẫn triển khai
├── HOSE_2020_2023_in.csv       # Bộ dữ liệu lịch sử giá & khối lượng HOSE 2020-2023
└── HOSE_5stocks_nhom5 (1).ipynb# Notebook Jupyter gốc phân tích chuyên sâu
```

---

## 🛠️ 7. Các Tính Năng Nổi Bật Trên Web App

- **Tab 1: 📊 Sàng Lọc & Phân Bổ Ngành:** Bảng xếp hạng định lượng Top 100 cổ phiếu trên tập Train, giải thích cơ sở phân bổ 5 nhóm ngành (BĐS, Hóa chất, Chứng khoán, Vận tải, Tiêu dùng).
- **Tab 2: 📈 Phân Tích Kỹ Thuật Từng Mã:** Biểu đồ Plotly tương tác 3 tầng (Giá kèm SMA Crossover & Điểm Mua/Bán; Khối lượng & OBV; Vị thế nắm giữ theo thời gian).
- **Tab 3: ⚖️ Tối Ưu Tỷ Trọng Danh Mục:** Ma trận tương quan lợi nhuận, biểu đồ so sánh phân bổ tỷ trọng Equal Weight (20% mỗi mã) vs MPT (Markowitz Max Sharpe).
- **Tab 4: 🏆 Kiểm Định Hiệu Suất Ngoài Mẫu:** Dashboard đối chiếu toàn diện 3 phương pháp trên năm 2022 kèm các chỉ số Alpha, Sharpe, Max Drawdown, Calmar Ratio.
- **Tab 5: 📉 Biểu Đồ Tăng Trưởng Vốn & Rủi Ro:** Đường cong NAV tăng trưởng vốn khởi điểm 1,000,000 VNĐ và biểu đồ sụt giảm tài sản (Underwater Chart).
- **Tab 6: 📑 Kết Luận Tự Động & Xuất Báo Cáo:** Trích xuất kết luận tự động dựa trên số liệu thực tế, 10 đề mục báo cáo Cao học chuẩn hóa và nút tải dữ liệu kết quả ra file CSV.

---

## 👨‍💻 8. Tác Giả & Bản Quyền

- **Học viên cao học:** Nhóm Nghiên cứu Định lượng Danh mục Đầu tư (CAO HỌC - HK3)
- **Nền tảng phát triển:** Python, Streamlit, Plotly, Scipy, TA-Lib/Ta, Hyperopt
- **Giấy phép:** MIT License (Tự do sử dụng và mở rộng cho mục đích học tập và nghiên cứu).
