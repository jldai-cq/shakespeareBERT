"""
by jl_dai 2024年12月25日
version 1
缺点：无法通过PoolerOutput获取句子的向量表示，因为AutoModelForMaskedLM中无池化层训练，需要手动添加池化层训练代码，详情见version 2
"""

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, BertTokenizer, \
    BertForMaskedLM
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import math


# 参数设置
BERT_PATH = "./model/bert_base_uncased"
model_name = "bert-base-uncased"
text_file = "./data/shakespeare_samples.txt"  # 包含所有数据的单一文件
max_seq_length = 512      # 输入最大token
out_model_path = "./model/shakespeareBert_v3"
train_ratio = 0.9  # 训练集比例
train_epochs = 20  # 训练轮次
batch_size = 16  # 批大小，建议设置为显存允许的最大值

# 加载分词器和预训练模型
tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
model = AutoModelForMaskedLM.from_pretrained(BERT_PATH).to('cuda')


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
    prediction_loss_only=True,
    logging_dir='./model/ShakespeareBert_v2/logs',  # 日志目录
    logging_strategy="steps",
    logging_steps=200,      # 1%到5%
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

# 保存最终模型
trainer.save_model(out_model_path)
tokenizer.save_vocabulary(out_model_path)

tokenizer.save_pretrained("你想要保存的路径")
model.save_pretrained("你想要保存的路径")

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


plt.plot(train_epoch, train_loss, label='Train Loss')
plt.plot(eval_epoch[:-1], eval_loss[:-1], label='Eval Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()