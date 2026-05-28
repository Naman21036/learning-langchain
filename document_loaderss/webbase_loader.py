from langchain_community.document_loaders import WebBaseLoader

url = "https://docs.n8n.io/?_gl=1*cz5klu*_gcl_au*MjY4NDY0ODgyLjE3NzkxODcwMjg.*_ga*MTIwNzI4NDcxOC4xNzc5MTg2OTQw*_ga_0SC4FF2FH9*czE3Nzk3OTQ5NjckbzQkZzAkdDE3Nzk3OTQ5NjckajYwJGwwJGgw"

loader = WebBaseLoader(url)
documents = loader.load()
print(documents[0].page_content)