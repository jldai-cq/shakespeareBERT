# 使用pdfplumber读取文件pdf_path = "./data/shakespeare.pdf"的第133页到1781页的数据，
# 并每200个words组成一个样本数据，每个样本数据单独一行，把所有分割的数据保存到txt文本中，和csv文件格式。
# 两种格式都要保存，csv有两列，第1列是id编号，第2列是每200个words数据。
# 大问题：它对于两栏pdf数据不友好，它居然直接忽略的两栏，直接将第1栏的句子拼接上了第2栏的句子，很离谱，

import pdfplumber
import csv
from tqdm import tqdm

# PDF 文件路径
pdf_path = "../data/shakespeare.pdf"

# 输出文件路径
output_txt_path = "../data/shakespeare_samples.txt"
output_csv_path = "./data/shakespeare_samples.csv"


# 提取指定页码范围内的文本
def extract_text_from_pdf(pdf_path, start_page, end_page):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num in tqdm(range(start_page - 1, end_page), desc="提取指定页码范围内的文本"):  # 页码从0开始
            text += pdf.pages[page_num].extract_text() + " "
    return text


# 分割文本为每200个单词一个样本
def split_text_into_samples(text, words_per_sample=200):
    words = text.split()
    samples = [" ".join(words[i:i + words_per_sample]) for i in tqdm(range(0, len(words), words_per_sample), desc="分割文本为每200个单词一个样本")]
    return samples


# 保存样本数据到TXT文件
def save_samples_to_txt(samples, output_path):
    with open(output_path, "w", encoding="utf-8") as txt_file:
        for sample in tqdm(samples, desc="保存样本数据到TXT文件"):
            txt_file.write(sample + "\n")


# 保存样本数据到CSV文件
def save_samples_to_csv(samples, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "text"])  # 写入表头
        for idx, sample in tqdm(enumerate(samples), desc="保存样本数据到CSV文件"):
            writer.writerow([idx + 1, sample])


# 主流程
if __name__ == "__main__":
    # 提取文本
    start_page = 134
    # end_page = 1782
    end_page = 140

    extracted_text = extract_text_from_pdf(pdf_path, start_page, end_page)

    # 分割为样本
    samples = split_text_into_samples(extracted_text, words_per_sample=200)

    # 保存到文件
    save_samples_to_txt(samples, output_txt_path)
    save_samples_to_csv(samples, output_csv_path)

    print(f"样本数据已保存到:\nTXT文件: {output_txt_path}\nCSV文件: {output_csv_path}")
