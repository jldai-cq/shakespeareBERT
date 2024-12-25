import torch
from transformers import  BertTokenizer, BertModel

BERT_PATH = "./model/test-model"

model = BertModel.from_pretrained(BERT_PATH)
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

# 输入文本
text = "Hello, how are you?"

# 编码文本
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# 获取模型的输出
with torch.no_grad():
    outputs = model(**inputs)


last_hidden_state = outputs.last_hidden_state
print("last_hidden_state: ", last_hidden_state)

# 提取 CLS token 的表示
cls_output = outputs.last_hidden_state[:, 0, :]
print("cls_output: ", cls_output)

pooler_output = outputs.pooler_output
print("pooler_output: ", pooler_output)

