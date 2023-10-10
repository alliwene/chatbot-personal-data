from typing import List
from dotenv import load_dotenv

load_dotenv()

from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# COHERE_API_KEY

cohere_embeddings = CohereEmbeddings()

# cohere_embeddings.embed_query('Veronica is a Pythonista')


def pretty_print_docs(docs: List[Document]) -> str:
    return f"\n{'-' * 100}\n".join(
        [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
    )


def load_db(file):
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # define embedding
    embeddings = CohereEmbeddings()
    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # similarity search
    result = db.similarity_search("Veronica is a Pythonista", k=2)

    return pretty_print_docs(result)