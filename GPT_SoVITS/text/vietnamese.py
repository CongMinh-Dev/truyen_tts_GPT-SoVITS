import re
import sys
import torch
import os
from transformers import AutoModelForMaskedLM, AutoTokenizer



# Chú ý: Đường dẫn phải chính xác
bert_path = "/content/GPT-SoVITS/pretrained_models/phobert-large" 

print(f">>> [DEBUG vietnamese.py] Đang nạp Tokenizer từ: {bert_path}"); sys.stdout.flush()
tokenizer = AutoTokenizer.from_pretrained(bert_path)
# --- THÊM CÁC DÒNG KIỂM TRA ký tự tách của phobert-large TẠI ĐÂY ---
print(">>> [KẾT QUẢ KIỂM TRA TOKENIZER]")
print(f"Special tokens map: {tokenizer.special_tokens_map}")
# Kiểm tra subword_prefix (Lưu ý: PhoBERT thường dùng suffix @@ nên prefix có thể là None hoặc rỗng)
if hasattr(tokenizer, 'subword_prefix'):
    print(f"Subword prefix: {tokenizer.subword_prefix}")
else:
    print("Tokenizer này không sử dụng subword_prefix (có thể dùng suffix @@)")

# Test thử một từ dễ bị tách sub-word
test_word = "khkũng"
print(f"Tokenize thử từ '{test_word}': {tokenizer.tokenize(test_word)}")
print("-" * 30)
sys.stdout.flush()
# -------------------------------------


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
    # Thêm khoảng trống quanh dấu câu để split() chuẩn hơn
    text = re.sub(r'([.,!?])', r' \1 ', text)
    # Loại bỏ các ký tự lạ, chỉ giữ lại chữ cái, số và dấu câu cơ bản
    text = re.sub(r'[^\w\s\.,!\?]', '', text, flags=re.UNICODE)
    # Xử lý khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    return text



def g2p(text):
    global tokenizer
    text = text_normalize(text)
    print(f"\n>>> [DEBUG G2P] Xử lý câu bằng PhoBERT Tokenizer: {text}")
    
    # Dùng PhoBERT để tách token
    raw_tokens = tokenizer.tokenize(text)
    
    words = []
    if len(raw_tokens) > 0:
        temp_word = raw_tokens[0]
        for i in range(1, len(raw_tokens)):
            if raw_tokens[i-1].endswith("@@"):
                # Nếu token trước có @@, gộp token hiện tại vào (xóa @@)
                temp_word = temp_word[:-2] + raw_tokens[i]
            else:
                words.append(temp_word)
                temp_word = raw_tokens[i]
        words.append(temp_word)

    all_phones = []
    word2ph = []
    for word in words:
        # Tách từng chữ cái làm phone (vẫn giữ logic đơn giản để tránh lỗi)
        current_word_phones = [char for char in word]

        all_phones.extend(current_word_phones)
        word2ph.append(len(current_word_phones))
    
    print(f">>> [DEBUG G2P] Kết quả sau gộp: {words}")
    print(f">>> [DEBUG G2P] Word2ph: {word2ph} | Tổng phones: {len(all_phones)}")
    sys.stdout.flush()
    return all_phones, word2ph


def get_bert_feature(text, word2ph):
    global bert_model, tokenizer, device
    text = text_normalize(text)
    pid = os.getpid()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        outputs = bert_model(**inputs, output_hidden_states=True)
        res = outputs.hidden_states[-1].squeeze(0).float().cpu().clone()
        token_list = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # Bỏ <s> và </s>
    useful_res = res[1:-1]
    useful_tokens = token_list[1:-1]
    
    # Gộp đặc trưng BERT theo quy tắc @@ y hệt như hàm g2p
    print(f"[BERT-P{pid}] Tokens thô từ PhoBERT: {useful_tokens}")
    word_signals = []
    if len(useful_tokens) > 0:
        temp_feat = useful_res[0]
        count = 1
        for i in range(1, len(useful_tokens)):
            if useful_tokens[i-1].endswith("@@"):
                temp_feat += useful_res[i]
                count += 1
            else:
                word_signals.append(temp_feat / count)
                temp_feat = useful_res[i]
                count = 1
        word_signals.append(temp_feat / count)

    # Chốt chặn kiểm tra: word_signals phải bằng len(word2ph)
    print(f"[BERT-P{pid}] Kiểm tra khớp: BERT có {len(word_signals)} từ | G2P có {len(word2ph)} từ")
    if len(word_signals) != len(word2ph):
        print(f"\033[91m[CẢNH BÁO LỆCH PHA] BERT {len(word_signals)} != G2P {len(word2ph)} tại câu: {text[:50]}\033[0m")
        final_word_signals = []
        for i in range(len(word2ph)):
            # Lấy vector BERT tương ứng với từ, nếu thiếu thì lấy cái cuối cùng
            if i < len(word_signals):
                final_word_signals.append(word_signals[i])
            else:
                final_word_signals.append(word_signals[-1] if len(word_signals) > 0 else torch.zeros_like(useful_res[0]))
    else:
        print(f"\033[92m[OK] Đã khớp hoàn toàn!\033[0m")
        final_word_signals = word_signals

    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_times = word2ph[i]
        feature = final_word_signals[i]
        for _ in range(repeat_times):
            phone_level_feature.append(feature)
    
    phone_level_feature = torch.stack(phone_level_feature, dim=0).T
    return phone_level_feature



