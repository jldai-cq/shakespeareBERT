"""
by jl_dai 2024年12月25日
version 2

自定义模型BertWithPoolerOutput:
    1.继承自 BertForMaskedLM
    2.forward：提取[CLS] token 表示（last_hidden_state[:, 0, :]），通过一个线性层（self.pooler）和 tanh 激活函数生成 pooler_output。
    3.最后，用 outputs._replace(pooler_output=pooler_output) 使得 BertForMaskedLM 的原始输出（MaskedLMOutput）增加了 pooler_output。

训练代码:
    将 model = BertForMaskedLM.from_pretrained(BERT_PATH) 修改为 model = BertWithPoolerOutput.from_pretrained(BERT_PATH)，加载自定义模型。
    数据集加载、分词器设置、训练设置、模型保存等保持不变
"""

import math
import torch
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling


class BertWithPoolerOutput(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        # 定义pooler层
        self.pooler = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = torch.nn.Tanh()
        # 使用BertModel来获得last_hidden_state
        self.bert = BertModel(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        # 使用BertModel来获取last_hidden_state
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)
        last_hidden_state = bert_outputs.last_hidden_state  # 获取last_hidden_state

        # 提取[CLS]的输出
        cls_token_representation = last_hidden_state[:, 0, :]  # 获取[CLS] token的表示
        # 通过池化层（全连接层）进行处理
        pooler_output = self.tanh(self.pooler(cls_token_representation))

        # 获取BERT的MaskedLM输出
        lm_outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask,
                                     token_type_ids=token_type_ids, labels=labels)

        # 返回字典，包含loss, logits, pooler_output
        return {
            'loss': lm_outputs.loss,
            'logits': lm_outputs.logits,
            'pooler_output': pooler_output
        }


# 参数设置
BERT_PATH = "./model/bert_base_uncased"
text_file = "./data/shakespeare_samples.txt"  # 包含所有数据的单一文件
max_seq_length = 512      # 输入最大token
out_model_path = "./model/shakespeareBert_v3"
train_ratio = 0.9  # 训练集比例
train_epochs = 10  # 训练轮次
batch_size = 8  # 批大小，建议设置为显存允许的最大值

# 加载分词器和预训练模型
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
model = BertWithPoolerOutput.from_pretrained(BERT_PATH).to('cuda')
# model = BertWithPoolerOutput.from_pretrained(BERT_PATH).to('cuda:0')


# 数据集加载和划分
# 加载整个数据文件
dataset = load_dataset("text", data_files={"text": text_file})["text"]

all_steps = train_epochs * len(dataset) / batch_size
print(f"all_steps: {all_steps}")

# 训练集和验证集划分
split_dataset = dataset.train_test_split(test_size=1 - train_ratio)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# 数据预处理：分词和截断
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
    )

# 显式分词
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 数据加载器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# 训练参数设置
training_args = TrainingArguments(
    output_dir=out_model_path,
    overwrite_output_dir=True,
    num_train_epochs=train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=2*batch_size,
    save_strategy="steps",
    save_steps=2000,        # save_steps必须是eval_steps的整数倍
    prediction_loss_only=True,  # 计算并返回模型的 预测损失，而不计算其他类型的指标（如准确率、F1分数等）
    logging_dir='./model/shakespeareBert_v3/logs',  # 日志目录
    logging_strategy="steps",
    logging_steps=500,      # 1%到5%
    learning_rate=5e-5,
    warmup_steps=1000,   # 取总训练steps的10%
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=1000,        # 5%到10%
    load_best_model_at_end=True,
    report_to="tensorboard",    # 可视化工具，用于监控和分析模型训练过程
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# 模型训练
trainer.train()

# 保存训练后的模型和分词器
trainer.save_model(out_model_path)
tokenizer.save_pretrained(out_model_path)

# 计算困惑度
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

# 绘制损失函数曲线图
# 打印 log_history 中的键名
log_history = trainer.state.log_history
# 初始化两个空列表来存储 loss 和 eval_loss 的值
train_loss = []
train_epoch = []
eval_loss = []
eval_epoch = []

# 遍历 log_history 列表
for entry in log_history:
    # 如果字典中有 'loss' 键，则将其值添加到 train_loss 列表中
    if 'loss' in entry:
        train_loss.append(entry['loss'])
        if 'epoch' in entry:
            train_epoch.append(entry['epoch'])
    # 如果字典中有 'eval_loss' 键，则将其值添加到 eval_loss 列表中
    if 'eval_loss' in entry:
        eval_loss.append(entry['eval_loss'])
        if 'epoch' in entry:
            eval_epoch.append(entry['epoch'])


plt.plot(train_epoch, train_loss, label='train_loss')
plt.plot(eval_epoch[:-1], eval_loss[:-1], label='eval_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
# 自动保存图像到文件而非显示出来
plt.savefig('./model/shakespeareBert_v3/loss_curve.png')
plt.close()