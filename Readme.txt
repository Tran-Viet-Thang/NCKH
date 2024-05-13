## CÁC FILE LIÊN QUAN
- data ban đầu của mô hình: https://drive.google.com/file/d/1G3PbamzwYyib68-n-mXDZcTP6MQTxc_c/view?usp=sharing
- code khi training phobert: https://colab.research.google.com/drive/1r-Dji3LjB_En5n_93-GNG6bnHqTdYcDH?usp=sharing
- code khi training xlm roberta base: https://colab.research.google.com/drive/1zN-gKkoc0OY8T_JTbHlu5Stq29ClTpSZ?usp=sharing

## HƯỚNG DẪN
Step 1: Setup environment: 
- python: 3.11
- java: jdk-11

Step 2: Install dependencies:
- Install library in requirement.txt

Step 3: Download model:
Với mô hình Phobert:
- Link model phobert: https://drive.google.com/file/d/12xMDi-JrK8yrvkTeAMoPw4aXkiMxVZjn/view?usp=sharing
- Sau khi tải mô hình về, copy mô hình vào thư mục "model"
- Đổi tên mô hình: model1.pth -> phobert.pth

Với mô hình xlm roberta base:
- Link model xlm roberta base: https://drive.google.com/file/d/1xL1O80H6DdlE2WrZ3aLOdPHwuYBRvb7A/view?usp=sharing
- Sau khi tải mô hình về, copy mô hình vào thư mục "model"
- Đổi tên mô hình: bart1.pth -> xlm_roberta_base.pth

Step 4:
- Vào file .env và đặt đường dẫn
  + VNCORENLP_PATH: absolute path của thư mục model/vncorenlp
  + PHOBERT_MODEL_PATH: absolute path của model phobert.pth
  + PHOBERT_MODEL_XLM_ROBERTA_BASE_PATH: absolute path của model xlm_roberta_base.pth

Step 5:
- Thực hiện chạy file app.py
- Click vào đường link (Ex: http://127.0.0.1:5000)
- Lựa chọn model và viết đoạn văn sau đó nhấn Send
- Đợi mô hình tạo câu hỏi

 

