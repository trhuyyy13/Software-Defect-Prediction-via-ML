# Software-Defect-Prediction-via-ML
# Software Defect Prediction 🛠️

## 🔗 Mô tả dự án  
Dự án tập trung vào việc phân tích và phân loại dữ liệu lỗi phần mềm từ các nguồn dữ liệu **NASA** và **PROMISE** bằng cách áp dụng các thuật toán **học máy**. Dữ liệu được tiền xử lý để đảm bảo chất lượng, bao gồm xử lý giá trị khuyết, lựa chọn đặc trưng quan trọng và cân bằng dữ liệu. Mục tiêu chính của dự án là phát triển một mô hình phân loại chính xác, giúp dự đoán lỗi phần mềm dựa trên dữ liệu lịch sử.

## 📚 Phương pháp tiếp cận  

### 1️⃣ Tiền xử lý dữ liệu  
- **Tích hợp dữ liệu**: Tải và hợp nhất dữ liệu từ nhiều tập tin CSV của **NASA** và **PROMISE**.  
- **Xử lý dữ liệu thiếu**: Áp dụng **KNN Imputation** để thay thế các giá trị bị thiếu.  
- **Chuẩn hóa dữ liệu**: Sử dụng **StandardScaler** để đưa tất cả các đặc trưng về cùng một khoảng giá trị, giúp mô hình hoạt động hiệu quả hơn.  
- **Lựa chọn đặc trưng**: Dùng **SelectKBest với ANOVA F-test** để xác định các đặc trưng quan trọng nhất, loại bỏ nhiễu.  
- **Cân bằng dữ liệu**: Sử dụng **SMOTE (Synthetic Minority Over-sampling Technique)** để khắc phục vấn đề mất cân bằng giữa các lớp dữ liệu.  

### 2️⃣ Huấn luyện mô hình  
- **Triển khai nhiều mô hình học máy**:  
  - **Random Forest**  
  - **Support Vector Machine (SVM)**  
  - **Logistic Regression**  
  - **Naive Bayes**  
- **Tăng cường hiệu suất mô hình** bằng cách kết hợp nhiều mô hình lại với nhau thông qua **Voting Classifier**.  
- **Tối ưu hóa tham số**: Sử dụng **StratifiedKFold** và kiểm tra chéo để điều chỉnh tham số của mô hình.  

### 3️⃣ Đánh giá mô hình  
- Sử dụng các chỉ số đo lường hiệu suất:  
  - **Accuracy (Độ chính xác)**  
  - **F1-score**  
  - **ROC-AUC (Diện tích dưới đường cong ROC)**  
  - **Geometric Mean Score** (Đánh giá khả năng mô hình dự đoán đúng các lớp dữ liệu mất cân bằng).  
- So sánh hiệu suất giữa các mô hình để chọn ra mô hình tối ưu nhất.  

## 🏆 Kết quả đạt được  
✅ **Xử lý thành công dữ liệu lớn** từ NASA và PROMISE.  
✅ **Cải thiện hiệu suất phân loại** bằng cách tối ưu hóa chọn lọc đặc trưng và cân bằng dữ liệu.  
✅ **Phát triển mô hình ensemble** kết hợp nhiều thuật toán giúp tăng độ chính xác dự đoán.  
✅ **Đạt điểm số phân loại cao**, chứng minh mô hình có độ tin cậy cao trong việc dự đoán lỗi phần mềm.  
✅ **Xác định các đặc trưng quan trọng nhất**, hỗ trợ các nhóm nghiên cứu và phát triển phần mềm trong việc tối ưu hóa quy trình kiểm thử.  

## 🚀 Ứng dụng thực tế  
Dự án này có thể giúp các tổ chức phần mềm phát hiện sớm lỗi phần mềm trong quá trình phát triển, từ đó giảm thiểu rủi ro và tối ưu hóa tài nguyên kiểm thử.  
