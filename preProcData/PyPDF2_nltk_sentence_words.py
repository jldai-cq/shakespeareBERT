"""
by jl_dai 2024年12月25日
工具：PyPDF2 + nltk
nltk：提取所有句子
分割方法：以句子为单位逐渐添加，直到超过words_per_sample停止添加
可选参数：
    start_page：pdf开始页码
    end_page：pdf结束页码
    words_per_sample：分割样本的单词数量
punkt保存讲解方法：
    GitHub下载nltk：https://github.com/nltk/nltk_data
    解压punkt复制到报错信息路径中任意一个即可
"""

import csv
from tqdm import tqdm
from PyPDF2 import PdfReader
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

# PDF 文件路径
pdf_path = "../data/shakespeare.pdf"

# 输出文件路径
output_txt_path = "../data/shakespeare_samples.txt"
output_csv_path = "./data/shakespeare_samples.csv"

# 提取指定页码范围内的文本
def extract_text_from_pdf(pdf_path, start_page, end_page):
    """
    从 PDF 提取指定页码范围的文本
    """
    text = ""
    reader = PdfReader(pdf_path)
    for page_num in tqdm(range(start_page - 1, end_page), desc="提取指定页码范围内的文本"):  # 页码从0开始
        page = reader.pages[page_num]
        text += page.extract_text() + " "
    return text


# 按句子分割并生成样本
def split_by_sentences(text, words_per_sample=200):
    """
    使用句子分割方法生成样本，每个样本约 200 个单词
    """
    sentences = sent_tokenize(text)
    samples = []
    current_sample = []
    current_word_count = 0

    for sentence in tqdm(sentences, desc="使用句子分割方法生成样本，每个样本约 200 个单词"):
        words_in_sentence = word_tokenize(sentence)
        if current_word_count + len(words_in_sentence) <= words_per_sample:
            current_sample.extend(words_in_sentence)
            current_word_count += len(words_in_sentence)
        else:
            samples.append(" ".join(current_sample))
            current_sample = words_in_sentence
            current_word_count = len(words_in_sentence)

    # 处理最后的样本
    if current_sample:
        samples.append(" ".join(current_sample))

    return samples


# 保存样本数据到TXT文件
def save_samples_to_txt(samples, output_path):
    """
    保存样本数据到 TXT 文件
    """
    with open(output_path, "w", encoding="utf-8") as txt_file:
        for sample in tqdm(samples, desc="保存样本数据到TXT文件"):
            txt_file.write(sample + "\n")


# 保存样本数据到CSV文件
def save_samples_to_csv(samples, output_path):
    """
    保存样本数据到 CSV 文件
    """
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "text"])  # 写入表头
        for idx, sample in tqdm(enumerate(samples), desc="保存样本数据到CSV文件"):
            writer.writerow([idx + 1, sample])


# 主流程
if __name__ == "__main__":
    # 页码范围
    start_page = 134
    end_page = 1782

    # 提取文本
    extracted_text = extract_text_from_pdf(pdf_path, start_page, end_page)

    # 按句子分割并生成样本
    samples = split_by_sentences(extracted_text, words_per_sample=200)

    # 保存到文件
    save_samples_to_txt(samples, output_txt_path)
    save_samples_to_csv(samples, output_csv_path)

    print(f"样本数据已保存到:\nTXT文件: {output_txt_path}\nCSV文件: {output_csv_path}")
