# 本代码需要text数据，该text可以是pypdf2已经把pdf数据加载并转换为了txt数据了
# 原理：首先每次添加一个句子，并判断添加句子后单词数量是否超过了 200
#      若没有超过继续添加下一个句子，否则直接结束并保存到样本数据集里面。

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

def split_by_sentences(text, words_per_sample=200):
    sentences = sent_tokenize(text)
    samples = []
    current_sample = []
    current_word_count = 0

    for sentence in sentences:
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
