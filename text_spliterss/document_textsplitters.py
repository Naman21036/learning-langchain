from langchain_text_splitters import RecursiveCharacterTextSplitter,Language
from langchain_community.document_loaders import TextLoader

loader= TextLoader("sample.txt")
documents= loader.load()

text_splitter= RecursiveCharacterTextSplitter.from_language(
    language= Language.PYTHON,
    chunk_size= 600,
    chunk_overlap=0
)
texts= text_splitter.split_documents(documents)
print(texts[1])
