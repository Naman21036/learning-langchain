from langchain_community.retrievers import WikipediaRetriever

retriever= WikipediaRetriever(top_k_results= 3, lang= "en")

query= "the geopolitical history of india and pakistan from the perspective of a chinese"

print(retriever.invoke(query))
