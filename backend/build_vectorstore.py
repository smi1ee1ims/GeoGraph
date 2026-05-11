"""
构建并保存向量库到磁盘
只需运行一次，之后直接加载
"""
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 获取当前目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(BASE_DIR, "data", "cleaned_湘教版地理必修一Chap2.md")
VECTORSTORE_PATH = os.path.join(BASE_DIR, "data", "vectorstore")

def build_and_save_vectorstore():
    """构建向量库并保存到磁盘"""
    print("正在加载文档...")
    loader = TextLoader(MD_PATH, encoding='utf-8')
    docs = loader.load()

    print("正在分割文档...")
    text_splitter = RecursiveCharacterTextSplitter(
        separators='\n\n',
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(docs)
    print(f"分割成 {len(splits)} 个片段")

    print("正在构建向量库（第一次需要下载模型，可能较慢）...")
    embeddings = HuggingFaceEmbeddings(
        model_name='all-MiniLM-L6-v2'
    )
    vectorstore = FAISS.from_documents(splits, embeddings)

    print(f"正在保存向量库到 {VECTORSTORE_PATH}...")
    vectorstore.save_local(VECTORSTORE_PATH)
    print("向量库构建并保存完成!")

def load_vectorstore():
    """从磁盘加载向量库"""
    if not os.path.exists(VECTORSTORE_PATH):
        print("向量库不存在，正在构建...")
        build_and_save_vectorstore()

    print("正在加载向量库...")
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    print("向量库加载完成!")
    return vectorstore

if __name__ == "__main__":
    build_and_save_vectorstore()
