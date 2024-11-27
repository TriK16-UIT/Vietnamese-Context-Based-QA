from transformers import pipeline

# Replace this with your own checkpoint
model_checkpoint = "checkpoints"
question_answerer = pipeline("question-answering", model=model_checkpoint, device="cuda")

context = "Năm 1871, Đức trở thành một quốc gia dân tộc khi hầu hết các quốc gia Đức thống nhất trong Đế quốc Đức do Phổ chi phối. Sau Chiến tranh thế giới thứ nhất và Cách mạng Đức 1918-1919, Đế quốc này bị thay thế bằng Cộng hòa Weimar theo chế độ nghị viện. Chế độ độc tài quốc xã được hình thành vào năm 1933, dẫn tới Chiến tranh thế giới thứ hai và một nạn diệt chủng. Sau một giai đoạn Đồng Minh chiếm đóng, hai nước Đức được thành lập: Cộng hòa Liên bang Đức và Cộng hòa Dân chủ Đức. Năm 1990, quốc gia được tái thống nhất."

question = "Cộng hòa Weimar chính thức thay thế đế quốc Đức kể từ sau sự kiện nào?"
out = question_answerer(question=question, context=context)

print(out)