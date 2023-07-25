from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0, "max_length": 500})

# Function to load all PDF files from a folder
def load_pdfs_from_folder(folder_path):
    pdf_files = [file for file in os.listdir(folder_path) if file.endswith(".pdf")]
    documents = []
    for pdf_file in pdf_files:
        loader = UnstructuredPDFLoader(os.path.join(folder_path, pdf_file))
        documents.extend(loader.load())
    return documents

# Folder path containing the PDF files
pdf_folder_path = "documents"
documents = load_pdfs_from_folder(pdf_folder_path)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()

docsearch = FAISS.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", return_source_documents=True, retriever=docsearch.as_retriever(search_type="similarity"))

while True:
    query = input("Enter your question (type 'exit' to quit): ")
    if query.lower() == "exit":
        break
    result = qa({"query": query})
    print(result)
