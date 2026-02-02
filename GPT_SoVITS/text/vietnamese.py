import re
import sys
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

print(">>> [DEBUG vietnamese.py] Đang nạp thư viện underthesea..."); sys.stdout.flush()
try:
    from underthesea import word_tokenize
    print(">>> [DEBUG vietnamese.py] Nạp underthesea thành công."); sys.stdout.flush()
except Exception as e:
    print(f">>> [DEBUG vietnamese.py] LỖI khi nạp underthesea: {e}"); sys.stdout.flush()

# Chú ý: Đường dẫn phải chính xác
bert_path = "/content/GPT-SoVITS/pretrained_models/phobert-large" 

print(f">>> [DEBUG vietnamese.py] Đang nạp Tokenizer từ: {bert_path}"); sys.stdout.flush()
tokenizer = AutoTokenizer.from_pretrained(bert_path)

print(f">>> [DEBUG vietnamese.py] Đang nạp PhoBERT Model (1024 dims)..."); sys.stdout.flush()
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)

# Kiểm tra GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f">>> [DEBUG vietnamese.py] Thiết bị sử dụng: {device}"); sys.stdout.flush()

if device == "cuda":
    print(">>> [DEBUG vietnamese.py] Đang chuyển Model sang GPU (half precision)..."); sys.stdout.flush()
    bert_model = bert_model.half().to(device)
else:
    print(">>> [DEBUG vietnamese.py] CẢNH BÁO: Đang chạy trên CPU, rất dễ bị Illegal Instruction!"); sys.stdout.flush()
    bert_model = bert_model.to(device)

print(">>> [DEBUG vietnamese.py] Nạp PhoBERT hoàn tất."); sys.stdout.flush()

def text_normalize(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s\.,!\?]', '', text, flags=re.UNICODE) 
    return text


def g2p(text):
    print(f">>> [DEBUG g2p] Đang tách từ chuẩn cho câu: {text}"); sys.stdout.flush()
    # Tách từ đơn giản bằng split để khớp với logic gộp của BERT
    words = text.split() 
    phones = []
    for word in words:
        # Giữ nguyên các ký tự để SoVITS giải mã âm thanh
        for char in word:
            phones.append(char)
        phones.append(" ") # Khoảng trắng phân tách từ
    
    print(f">>> [DEBUG g2p] Tách từ thành công. Số lượng phones: {len(phones)}"); sys.stdout.flush()
    return phones



def get_bert_feature(text, word2ph):
    print(f"\n[CHECK] Văn bản: {text}")
    print(f"[CHECK] word2ph (Số phone mỗi từ): {word2ph} | Tổng số từ G2P đếm: {len(word2ph)}")
    sys.stdout.flush()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt").to(device)
        outputs = bert_model(**inputs, output_hidden_states=True)
        res = outputs.hidden_states[-1].squeeze(0)
        res = res.float().cpu()

    # 1. Kiểm tra Tokenizer thực tế của PhoBERT
    token_list = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    print(f">>> [DEBUG BERT] Tokens thực tế PhoBERT: {token_list}")
    
    # Bỏ [CLS] và [SEP]
    useful_res = res[1:-1]
    useful_tokens = token_list[1:-1]

    # 2. Xử lý gộp Sub-tokens (@@)
    word_signals = []
    if len(useful_tokens) > 0:
        temp_feat = useful_res[0]
        count = 1
        current_word_tokens = [useful_tokens[0]]
        
        for i in range(1, len(useful_tokens)):
            if useful_tokens[i-1].endswith("@@"):
                temp_feat += useful_res[i]
                count += 1
                current_word_tokens.append(useful_tokens[i])
            else:
                word_signals.append(temp_feat / count)
                # print(f"    + Đã gộp nhóm: {current_word_tokens}") # Bật nếu muốn xem chi tiết gộp
                temp_feat = useful_res[i]
                count = 1
                current_word_tokens = [useful_tokens[i]]
        word_signals.append(temp_feat / count)

    print(f">>> [DEBUG BERT] Số lượng từ sau khi gộp: {len(word_signals)}")
    
    # 3. KIỂM TRA ĐỘ LỆCH (Đây là chỗ dễ sai nhất)
    if len(word_signals) != len(word2ph):
        print(f"!!! CẢNH BÁO LỆCH PHA: PhoBERT thấy {len(word_signals)} từ, nhưng word2ph yêu cầu {len(word2ph)} từ.")
        print(f"    -> Điều này sẽ khiến AI phát âm như tiếng nước ngoài.")
    sys.stdout.flush()

    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_times = word2ph[i]
        idx = min(i, len(word_signals) - 1)
        feature = word_signals[idx] if word_signals else res[0]
        
        for _ in range(repeat_times):
            phone_level_feature.append(feature)

    return torch.stack(phone_level_feature, dim=0).T


