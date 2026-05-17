from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

with open("structured_output/review.txt", "r") as file:
    review_text= file.read()

llm = ChatGroq(
    model="llama-3.1-8b-instant"
)


class Review(BaseModel):
    summary: str = Field(description="A brief summary of the review")
    sentiment: str = Field(description="A rating which can be positive, negative, or neutral")
    key_topics: list[str] = Field(description="A list of key topics mentioned in the review")

structured_model= llm.with_structured_output(Review)


response= structured_model.invoke(review_text)

print(response)
print("Summary:", response.summary)
print("Sentiment:", response.sentiment)
print("Key Topics:", response.key_topics)

