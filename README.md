# Dịch Máy Tiếng Việt - Tiếng Lào

## Tổng quan
Dự án này phát triển một mô hình dịch máy tự động cho cặp ngôn ngữ ít tài nguyên tiếng Việt - tiếng Lào, sử dụng kiến trúc **Transformer** hiện đại. Mô hình được huấn luyện từ đầu, tích hợp kỹ thuật **RoPE (Rotary Positional Embedding)** và mã hóa **Byte Pair Encoding (BPE)** để xử lý hiệu quả hai ngôn ngữ có cấu trúc khác biệt. Dự án cũng xây dựng một tập dữ liệu song ngữ mới với khoảng **130,000 cặp câu**.

## Đặc điểm chính
* **Kiến trúc**: Transformer với 6 tầng encoder/decoder, 8 đầu attention, tích hợp RoPE.
* **Tokenizer**: Hai tokenizer BPE riêng biệt cho tiếng Việt và tiếng Lào, mỗi cái có từ vựng 16,000 đơn vị.
* **Dữ liệu**: 130,000 cặp câu song ngữ.
* **Huấn luyện**: Sử dụng Adam optimizer, label smoothing, và lịch trình học với warm-up (4000 bước).
* **Suy luận**: Tìm kiếm chùm (beam search) với độ rộng 3, độ dài tối đa 100.

## Tác giả
* Nguyễn Đức Hùng - 22021109
* Nguyễn Đức Hiển - 22028178
* Lôi Đình Nhất - 22021152
