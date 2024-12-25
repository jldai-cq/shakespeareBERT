"""
author: jl_dai
tiem: 2024年12月25日
"""

import PyPDF2

# 使用open的‘rb’方法打开pdf文件，使用二进制模式
mypdf = open('../data/shakespeare.pdf', mode='rb')

# 调用PdfReader函数
pdf_document = PyPDF2.PdfReader(mypdf)

# 获取PDF文档的页数
page_num = len(pdf_document.pages)
print(f"all page num: {page_num}")  # 2038 pages

# 调用PdfReader对象的pages()方法，传入页码，取得Page对象：输出PDF文档的第一页内容
page = pdf_document.pages[133]    # 2(133)-1652(1781)
print(f"page: {page}")

# 调用Page对象的extract_text()方法，返回该页文本的字符串
text = page.extract_text()
print(f"text: {text}")

import PyPDF2

# 使用open的‘rb’方法打开pdf文件，使用二进制模式
mypdf = open('./data/shakespeare.pdf', mode='rb')

# 调用PdfReader函数
pdf_document = PyPDF2.PdfReader(mypdf)

# 获取PDF文档的页数
page_num = len(pdf_document.pages)
print(f"all page num: {page_num}")  # 2038 pages

# 调用PdfReader对象的pages()方法，传入页码，取得Page对象：输出PDF文档的第一页内容
page = pdf_document.pages[133]    # 2(133)-1652(1781)
print(f"page: {page}")

# 调用Page对象的extract_text()方法，返回该页文本的字符串
text = page.extract_text()
print(f"text: {text}")