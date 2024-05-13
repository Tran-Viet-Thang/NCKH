from model.xlm_roberta_base import XLMRobertaBase

model = XLMRobertaBase()
paragragh = "Ba Bể có những dãy núi đá vôi dựng đứng, nhiều thung lũng lớn và rừng cây xanh rì bao bọc là điểm đến lý tưởng cho du khách mê phiêu lưu, khám phá. Các thác nước, hang động, hồ ở đây tạo nên một khung cảnh thiên nhiên kỳ thú với hơn 550 loài động thực vật quý hiếm. Khám phá Ba Bể bằng thuyền hay trekking hoặc đạp xe xuyên rừng chắc chắn sẽ cho bạn nhiều trải nghiệm đáng nhớ, sau đó hãy nghỉ ngơi, nạp năng lượng ở những căn homestay, nhà nghỉ trong các bản Tày địa phương."
questions = model.generate_question(paragragh)
print(questions)