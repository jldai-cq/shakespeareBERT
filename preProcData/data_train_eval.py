from datasets import load_dataset, Dataset

text_file = "../data/shakespeare_samples.txt"  # 包含所有数据的单一文件
train_ratio = 0.9  # 训练集比例

# 数据集加载和划分
# 加载整个数据文件
dataset = load_dataset("text", data_files={"text": text_file})["text"]
print(f"dataset: {len(dataset)}")

# 获取某部分dataset数据
# dataset = dataset.select(range(5))

# 训练集和验证集划分
split_dataset = dataset.train_test_split(test_size=1 - train_ratio)
train_dataset = split_dataset["train"]
print(f"train: {len(train_dataset)}")
eval_dataset = split_dataset["test"]
print(f"eval: {len(eval_dataset)}")








