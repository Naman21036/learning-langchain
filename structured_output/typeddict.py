from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

with open("structured_output/review.txt", "r") as file:
    review_text= file.read()

llm= HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    temperature= 1.4
)

class review(TypedDict):
    summary: Annotated["str", "A brief summary of the review"]
    sentiment: Annotated["str", "A rating which can be positive, negative, or neutral"]
    key_topics: Annotated["list[str]", "A list of key topics mentioned in the review"]

model= ChatHuggingFace(llm=llm)
structured_model= model.with_structured_output(review)


response= structured_model.invoke(review_text)

print(response)
print("Summary:", response["summary"])
print("Sentiment:", response["sentiment"])
print("Key Topics:", response["key_topics"])
