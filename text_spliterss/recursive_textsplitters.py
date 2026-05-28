from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader= TextLoader("structured_output\\review.txt")
documents= loader.load()

text_splitter= RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
chunk= text_splitter.split_text(documents[0].page_content)
print(chunk)
