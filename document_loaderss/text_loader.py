from langchain_community.document_loaders import TextLoader

loader= TextLoader("structured_output\\review.txt")
documents= loader.load()
print(documents[0].page_content)