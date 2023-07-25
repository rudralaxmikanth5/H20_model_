
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

class DocumentRetrievalQA:
    def __init__(self, pdf_folder_path, huggingface_token, model_repo_id):
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_token
        self.llm = HuggingFaceHub(repo_id=model_repo_id, model_kwargs={"temperature": 0, "max_length": 500})
        self.documents = self._load_pdfs_from_folder(pdf_folder_path)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        self.texts = self.text_splitter.split_documents(self.documents)
        self.embeddings = HuggingFaceEmbeddings()
        self.docsearch = FAISS.from_documents(self.texts, self.embeddings)
        self.qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="refine", return_source_documents=True, retriever=self.docsearch.as_retriever(search_type="similarity"))

    def _load_pdfs_from_folder(self, folder_path):
        pdf_files = [file for file in os.listdir(folder_path) if file.endswith(".pdf")]
        documents = []
        for pdf_file in pdf_files:
            loader = UnstructuredPDFLoader(os.path.join(folder_path, pdf_file))
            documents.extend(loader.load())
        return documents

    def ask_question(self, question):
        result = self.qa({"query": question})
        return result
