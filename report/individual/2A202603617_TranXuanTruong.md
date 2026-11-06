# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

SEED: 7668

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | Có |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | Có |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | Có |

# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác

- **Bài toán:** Phân loại tin nhắn SMS thành Spam và Ham.
- **LLM Provider:** Gemini.
- **LLM Model:** gemini-3.1-flash-lite.
- **Nguồn:** Kết quả thực nghiệm và output trong notebook Colab.
- **Báo cáo:** Tổng hợp với sự hỗ trợ của AI. Các số liệu thực nghiệm được giữ theo kết quả đã chạy.


Gemini API đã phản hồi thành công ở lần kiểm tra. Model Gemini 3.6 Flash gặp lỗi HTTP 429 do giới hạn quota nên không có kết quả thực nghiệm để so sánh với Gemini 3.1 Flash Lite.

## 2. Kết quả chính

### 2.1. Kết quả các mô hình

Bảng dưới đây tổng hợp F1-score trên tập test thông thường và SUB theo kết quả đã gửi.

| Phương án | F1 test | F1 SUB |
|---|---:|---:|
| BoW + LogReg | 0.889 | 0.947 |
| BoW + MLP | 0.919 | 1.000 |
| Embedding + kNN | 1.000 | 1.000 |
| Embedding + LogReg | 0.974 | 1.000 |
| Embedding + MLP head | 1.000 | 1.000 |
| Gemini 3.1 Flash Lite Zero-shot | Chưa đo toàn bộ test | 0.947 |
| Gemini 3.1 Flash Lite Few-shot | Chưa đo toàn bộ test | 1.000 |

LLM chỉ được đánh giá trên SUB nên không so sánh trực tiếp F1 SUB với F1 test của các mô hình truyền thống như thể chúng được đo trên cùng một tập dữ liệu.

### 2.2. Kết quả LLM

| Chỉ số | Zero-shot | Few-shot |
|---|---:|---:|
| Model | Gemini 3.1 Flash Lite | Gemini 3.1 Flash Lite |
| Số ví dụ mẫu | 0 | 4 |
| F1 SUB | 0.947 | 1.000 |
| Throughput (tin/s) | 0.44 | 0.30 |
| p50 latency (ms) | 725.95 | 782.11 |
| p95 latency (ms) | 6,880 | 16,108.84 |
| Chi phí (USD) | 0.00113 | 0.00259 |
| Invalid | 0 | 0 |

Few-shot tăng F1 từ 0.947 lên 1.000 trên SUB. Chi phí tăng từ $0.00113 lên $0.00259, tương đương khoảng 2.29 lần.

Kết quả này cho thấy 4 ví dụ mẫu từ TRAIN có thể giúp LLM phân loại tốt hơn trên tập SUB đang kiểm tra. Tuy nhiên, SUB có quy mô nhỏ nên chưa thể khẳng định khả năng tổng quát hóa trên dữ liệu thực tế.

## 3. Dự đoán và thực tế

### 3.1. Dự đoán về mô hình truyền thống

Dự đoán ban đầu là MLP có thể đạt kết quả tốt hơn Logistic Regression nhờ khả năng học các mối quan hệ phức tạp giữa các đặc trưng.

Kết quả thực tế:

- BoW + MLP đạt F1 = 0.919, cao hơn BoW + LogReg với F1 = 0.889.
- Embedding + MLP head đạt F1 = 1.000 trên SUB.
- Embedding + LogReg đạt F1 = 1.000 trên SUB.

Nhìn chung, kết quả phù hợp với dự đoán rằng MLP có thể khai thác thêm thông tin từ các đặc trưng. Tuy nhiên, chưa thể kết luận MLP luôn tốt hơn Logistic Regression.

### 3.2. Dự đoán về LLM

Dự đoán ban đầu là LLM có khả năng xử lý ngữ cảnh tốt hơn mô hình truyền thống. Few-shot có thể cải thiện kết quả nhưng sẽ làm tăng chi phí.

Kết quả thực tế:

- Gemini Zero-shot đạt F1 = 0.947.
- Gemini Few-shot đạt F1 = 1.000.
- Few-shot có chi phí khoảng 2.29 lần Zero-shot.
- Cả hai phương án đều có invalid = 0.

Kết quả phù hợp với dự đoán về hiệu quả của Few-shot. Tuy nhiên, chưa có kết quả thực nghiệm của Gemini 3.6 Flash để so sánh với Gemini 3.1 Flash Lite.

## 4. Phân tích Stress Test

### 4.1. Kết quả trên 50 trường hợp khó

| Model | F1 SUB | F1 50 case | Accuracy 50 case |
|---|---:|---:|---:|
| BoW + LogReg | 0.947 | 0.286 | 0.50 |
| BoW + MLP | 1.000 | 0.333 | 0.52 |
| Embedding + kNN | 1.000 | 0.717 | 0.70 |
| Embedding + LogReg | 1.000 | 0.640 | 0.64 |
| Embedding + MLP head | 1.000 | 0.680 | 0.68 |
| Gemini Zero-shot | 0.947 | 1.000 | 1.00 |
| Gemini Defended Prompt | 0.947 | 0.982 | 0.98 |

*Lưu ý: Mốc F1 SUB của Defended Prompt được đối chiếu theo kết quả Zero-shot. Không xem đây là kết quả đo độc lập của prompt phòng thủ trên SUB.*

### 4.2. Phân tích theo nhóm trường hợp

| Nhóm trường hợp | BoW LR | BoW MLP | Emb kNN | Emb LR | Emb MLP | Gemini Zero-shot | Defended Prompt |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tưởng spam nhưng không phải | 100% | 90% | 70% | 80% | 90% | 100% | 100% |
| Thao túng/lừa đảo | 12% | 25% | 88% | 38% | 50% | 100% | 100% |
| Spam giọng tự nhiên | 0% | 10% | 30% | 30% | 30% | 100% | 90% |
| Né từ khóa | 17% | 0% | 83% | 83% | 83% | 100% | 100% |
| Prompt injection | 67% | 67% | 83% | 100% | 100% | 100% | 100% |
| Nói về spam | 100% | 100% | 50% | 50% | 50% | 100% | 100% |
| Tin giao dịch hợp lệ | 83% | 100% | 100% | 83% | 83% | 100% | 100% |

BoW + MLP giảm mạnh từ F1 SUB = 1.000 xuống F1 stress test = 0.333. Điều này cho thấy mô hình gặp khó khăn khi dữ liệu có cách diễn đạt khác với những gì mô hình đã học.

BoW chủ yếu dựa trên sự xuất hiện của từ khóa nên dễ bỏ sót spam có cách viết tự nhiên hoặc cố tình né từ khóa. Embedding cải thiện khả năng nhận biết ngữ nghĩa nhưng vẫn gặp khó khăn với những trường hợp cần hiểu ngữ cảnh sâu.

Gemini Zero-shot đạt kết quả tốt nhất trên bộ 50 case. Tuy nhiên, đây là bộ dữ liệu kiểm thử được xây dựng để thử thách mô hình chứ không đại diện cho tỷ lệ spam thực tế.

### 4.3. Nhận xét về Prompt Injection

Gemini Zero-shot và Defended Prompt đều đạt 100% accuracy trên nhóm Prompt Injection.

Defended Prompt không cải thiện thêm kết quả ở nhóm này. Tuy nhiên, accuracy của nhóm Spam giọng tự nhiên giảm từ 100% xuống 90%.

Điều này cho thấy việc bổ sung hướng dẫn phòng thủ không nhất thiết cải thiện tất cả nhóm dữ liệu. Cần kiểm tra thêm trên tập dữ liệu lớn và đa dạng hơn.

## 5. Hand Calculation

Các phép tính được thực hiện trên giấy và làm tròn 2 chữ số thập phân.

| Đại lượng | Kết quả |
|---|---:|
| Baseline Accuracy | Cần đối chiếu số lượng Ham và tổng số SMS sau khi bỏ trùng |
| Sigmoid(0) | 0.50 |
| F1 với TP = 80, FP = 20, FN = 40 | 0.73 |
| Cosine Similarity giữa [1, 2, 0] và [2, 1, 2] | 0.60 |
| Chi phí LLM cho 1 triệu SMS/ngày theo đề bài | $10.50/ngày |

Công thức F1:

F1 = 2 × Precision × Recall / (Precision + Recall)

Công thức Sigmoid:

σ(x) = 1 / (1 + e^(-x))

Công thức Cosine Similarity:

Cosine(A, B) = (A · B) / (||A|| × ||B||)

Chi phí LLM được tính từ 100 token input và 1 token output cho mỗi SMS. Giá lần lượt là $0.10 và $0.50 trên 1 triệu token.

## 6. Phương án cho sản phẩm thực tế

### 6.1. So sánh các phương án

Kết quả cho thấy mô hình truyền thống có ưu điểm về khả năng xử lý nhanh và không phát sinh chi phí API cho mỗi tin nhắn. Tuy nhiên, các mô hình BoW gặp khó khăn rõ rệt trên bộ stress test.

Embedding cải thiện khả năng nhận biết ngữ nghĩa và đạt kết quả tốt hơn BoW trên nhiều trường hợp khó.

LLM có kết quả tốt nhất trên bộ stress test đã xây dựng. Tuy nhiên, độ trễ của Gemini 3.1 Flash Lite tương đối cao:

- Zero-shot có throughput khoảng 0.44 tin/s.
- Few-shot có throughput khoảng 0.30 tin/s.
- p95 latency lần lượt khoảng 6.88 giây và 16.11 giây.

Vì vậy, không nên sử dụng LLM để xử lý toàn bộ SMS theo cách gọi API tuần tự nếu hệ thống cần phản hồi nhanh và có lưu lượng lớn.

### 6.2. Đề xuất triển khai

Có thể xây dựng hệ thống phân loại nhiều tầng:

1. Sử dụng mô hình truyền thống để xử lý những tin nhắn có đặc điểm rõ ràng.
2. Sử dụng Embedding để cải thiện khả năng nhận diện những tin nhắn có cách diễn đạt khác nhau.
3. Chuyển các trường hợp không chắc chắn hoặc có dấu hiệu lừa đảo sang LLM để phân tích thêm.
4. Kết hợp kiểm tra URL, số điện thoại và các dấu hiệu bất thường để tăng độ an toàn.

Không nên chỉ dựa vào xác suất dự đoán của một mô hình vì mô hình vẫn có thể tự tin nhưng dự đoán sai.

Trước khi triển khai thực tế cần kiểm tra thêm trên dữ liệu SMS mới và đo hiệu năng khi có nhiều yêu cầu đồng thời.

## 7. AI Log và điều đã học

| Phần | Bạn hỏi AI gì | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| 0 | Data Leakage là gì? Data Cleaning và loại bỏ dữ liệu trùng lặp có ý nghĩa gì? | Giải thích Data Leakage và tầm quan trọng của việc làm sạch dữ liệu trước khi huấn luyện mô hình. | Đối chiếu với quy trình xử lý dữ liệu trong bài. |
| 1 | Baseline, Accuracy và cách đánh giá mô hình luôn dự đoán ham là gì? | Giải thích khái niệm và công thức Accuracy trong bài toán phân loại SMS. | Đối chiếu công thức với dữ liệu sau khi bỏ trùng. |
| 2 | Bag of Words, TF-IDF, Embedding và Logistic Regression là gì? | Giải thích cách biểu diễn văn bản thành vector và nguyên lý hoạt động của các mô hình phân loại. | So sánh ưu nhược điểm và đối chiếu kết quả thực nghiệm. |
| 3 | Precision, Recall, F1-score, Confusion Matrix và Cosine Similarity là gì? | Giải thích các chỉ số đánh giá mô hình và cách đo độ tương đồng giữa hai vector. | Tự kiểm tra công thức và tính lại các ví dụ trong đề. |
| 4 | Zero-shot, Few-shot, Prompt Engineering, Token và LLM là gì? Hướng dẫn viết và điều chỉnh code gọi Gemini API để phân loại SMS. | Giải thích cách sử dụng LLM. Hỗ trợ viết và chỉnh sửa code cấu hình API, hàm `call_llm()`, `build_classification_prompt()`, `parse_label()` và `llm_classify_batch()`. Hướng dẫn xử lý lỗi API, đo latency và ước tính chi phí. | Đọc lại code và điều chỉnh cấu hình Gemini. Kiểm tra API key bằng Colab Secrets. Chạy thử API và kiểm tra kết quả phân loại. |
| 5 | Sigmoid là gì? Cách tính F1-score, Cosine Similarity và chi phí sử dụng LLM như thế nào? | Giải thích các khái niệm và công thức toán học liên quan. | Tự thực hiện phép tính trên giấy và kiểm tra lại kết quả. |

### Điều đã học

Qua bài thực hành, tôi hiểu rõ hơn sự khác biệt giữa các phương pháp phân loại văn bản truyền thống, Embedding và LLM.

Accuracy không phải lúc nào cũng phản ánh đầy đủ chất lượng mô hình. Precision, Recall và F1-score giúp đánh giá cụ thể hơn khả năng nhận diện spam.

BoW có tốc độ xử lý tốt nhưng phụ thuộc nhiều vào từ khóa. Embedding có khả năng biểu diễn ngữ nghĩa tốt hơn. LLM có thể hiểu ngữ cảnh và xử lý những trường hợp khó nhưng cần cân nhắc độ trễ và chi phí API.

Tôi cũng hiểu rằng kết quả tốt trên tập test thông thường chưa đảm bảo mô hình hoạt động tốt trên dữ liệu thực tế. Stress test giúp phát hiện những điểm yếu mà tập test thông thường có thể chưa thể hiện được.

Trong quá trình làm bài, tôi có sử dụng AI để tham khảo lý thuyết và hỗ trợ code ở Phần 4. Tôi đã đọc lại, điều chỉnh cấu hình và kiểm tra kết quả thực thi thay vì chỉ sử dụng nguyên kết quả do AI cung cấp.
