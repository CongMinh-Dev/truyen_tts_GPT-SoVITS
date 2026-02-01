import re
from underthesea import word_tokenize

# Bảng ký hiệu âm đọc cơ bản để AI không bị lỗi
# Bạn có thể mở rộng bảng này nếu muốn độ chính xác cao hơn
def text_normalize(text):
    # Không dùng regex để xóa ký tự nữa vì nó đang xóa nhầm chữ tiếng Việt
    # Chỉ viết thường và xóa khoảng trắng thừa ở đầu/cuối
    text = text.lower().strip()
    # Nếu muốn xóa các ký tự lạ mà vẫn giữ dấu tiếng Việt, dùng cách này:
    text = re.sub(r'[^\w\s\.,!\?]', '', text, flags=re.UNICODE) 
    return text

def g2p(text):
    # Sử dụng underthesea để tách từ đúng ngữ pháp tiếng Việt
    words = word_tokenize(text)
    phones = []
    for word in words:
        # Trong cấu hình đơn giản, chúng ta coi mỗi chữ cái là một âm (phone)
        # GPT-SoVITS sẽ tự học cách đọc các chữ cái này qua quá trình Train
        for char in word:
            if char.strip():
                phones.append(char)
        phones.append(" ") # Khoảng trắng giữa các từ
    return phones


import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Chú ý: Thay đường dẫn này bằng đường dẫn thực tế đến folder PhoBERT trên Colab của bạn
bert_path = "/content/GPT-SoVITS/pretrained_models/phobert-large" 

tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)

# Kiểm tra xem có GPU không để chạy cho nhanh
device = "cuda" if torch.cuda.is_available() else "cpu"
bert_model = bert_model.to(device)

def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt").to(device)
        outputs = bert_model(**inputs, output_hidden_states=True)
        # Lấy hidden states cuối cùng (1024 chiều cho bản large)
        res = outputs.hidden_states[-1].squeeze(0)
        
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_times = word2ph[i]
        # PhoBERT dùng token [CLS] ở vị trí 0, nên từ đầu tiên bắt đầu từ index 1
        if i + 1 < res.shape[0]:
            feature_at_word = res[i + 1]
        else:
            feature_at_word = res[-1] # Tránh lỗi index out of range
            
        for _ in range(repeat_times):
            phone_level_feature.append(feature_at_word)
            
    return torch.stack(phone_level_feature).T.cpu()