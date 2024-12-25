import torch
import matplotlib
matplotlib.use('TkAgg')
from datasets import load_dataset
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModelForMaskedLM, AutoModel
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

text_file = "./data/history_work/history_data_20words.txt"
BERT_PATH = "./model/shakespeareBert_v3"

# 加载二次预训练模型和分词器
tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
model = AutoModel.from_pretrained(BERT_PATH).eval().to("cuda")

# 定义一个数据集类，用于支持 DataLoader
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]['text']

# 修改 get_bert_embeddings 函数，支持批量处理
def get_bert_embeddings(data, model, tokenizer, batch_size=16, max_length=256):
    device = next(model.parameters()).device  # 获取模型所在设备
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    embeddings = []

    with torch.no_grad():
        for batch_texts in tqdm(dataloader, desc="model to CLS"):
            inputs = tokenizer(
                list(batch_texts),
                return_tensors='pt',
                padding="max_length",
                truncation=True,
                max_length=max_length
            ).to(device)  # 移动到 GPU
            outputs = model(**inputs)
            pooler_output = outputs.pooler_output  # [CLS] 表示
            embeddings.append(pooler_output.cpu())  # 输出的 pooler_output 移动到 CPU，避免 GPU 显存不足

    return torch.cat(embeddings).numpy()

# 使用 DataLoader 加载数据
dataset = load_dataset("text", data_files={"text": text_file})["text"]
# dataset = dataset.select(range(200))
text_dataset = TextDataset(dataset)

# 获取 [CLS] 嵌入向量
embeddings = get_bert_embeddings(text_dataset, model, tokenizer, batch_size=256)
# print(embeddings)

# 聚类
n_clusters = 5  # 假设我们想要聚类成2个类别
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)

# 聚类结果
cluster_labels = kmeans.labels_

# 输出聚类结果
# print("Cluster labels:", cluster_labels)

# 将结果保存到文件
with open("./data/history_work/cluster_results.txt", "w", encoding="utf-8") as f:
    for i, text in tqdm(enumerate(dataset), desc="cluster_results write to txt"):
        f.write(f"Text {i} -> Cluster {cluster_labels[i]} : {text['text']} \n")


# 聚类结果可视化, PCA降维到2维
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# 可视化
plt.figure(figsize=(10, 7))
for cluster in tqdm(range(n_clusters), desc="process KMeans Clustering Visualization"):
    cluster_points = reduced_embeddings[cluster_labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}")

plt.title("KMeans Clustering Visualization (PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()
