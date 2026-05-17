from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    temperature= 1.4
)

model= ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")
    city: str = Field(description="The city where the person lives")

parser= PydanticOutputParser(pydantic_object=Person)

template= PromptTemplate(
    template= "Generate a person's information in the following format: {format_instructions}",
    partial_variables= {"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser
response = chain.invoke({})
print(response)


