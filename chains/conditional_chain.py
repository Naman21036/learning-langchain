from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm1= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature= 1.4
)
class ReviewSentiment(BaseModel):
    sentiment : Literal["positive", "negative", "neutral"] = Field(description="The sentiment of the review")

parser2= PydanticOutputParser(
    pydantic_object=ReviewSentiment
)
model1= ChatHuggingFace(llm=llm1)
parser1= StrOutputParser()

prompt= PromptTemplate(
    template="You are a helpful assistant. Please analyze the review given by the customer and return the sentiment:\n {text}\n {format_instructions}",
    input_variables=["text"],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)

sentiment_chain= prompt | model1 | parser2

prompt2= PromptTemplate(
    template= "You are experised sales person with 20 years of experience. Please generate a feedback on basis of the following positive review: {text}",
    input_variables=["text"]
)

prompt3= PromptTemplate(
    template= "You are experised sales person with 20 years of experience. Please generate a  feedback on basis of the following negativereview: {text}",
    input_variables=["text"]
)

branch_chain= RunnableBranch(
    (lambda x: x.sentiment== "positive", prompt2 | model1 | parser1),
    (lambda x: x.sentiment== "negative", prompt3 | model1 | parser1),
    RunnableLambda(lambda x: "Can't determine sentiment")
)

chain= sentiment_chain | branch_chain
response= chain.invoke({"text": "The product quality is really bad and I am not satisfied with the purchase"})
print(response)

chain.get_graph().print_ascii()