from langchain_community.document_loaders import PyPDFLoader

loader= PyPDFLoader("example.pdf")
documents= loader.load()
print(documents)