import re
from underthesea import word_tokenize

# Bảng ký hiệu âm đọc cơ bản để AI không bị lỗi
# Bạn có thể mở rộng bảng này nếu muốn độ chính xác cao hơn
def text_normalize(text):
    text = text.lower()
    # Loại bỏ các ký tự đặc biệt không cần thiết
    text = re.sub(r'[^\w\s\.,!\?]', '', text)
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

# GPT-SoVITS yêu cầu module phải có hàm này
def get_bert_feature(text, word2ph):
    # Hàm này thường được gọi từ 1-get-text.py
    # Nếu bạn đã sửa 1-get-text.py như tôi hướng dẫn trước đó thì hàm này ở đây có thể để trống
    pass