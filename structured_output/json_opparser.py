from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    temperature= 1.4
)

model= ChatHuggingFace(llm=llm)
parser= JsonOutputParser()

template= PromptTemplate(
    template= "Give the finance capital, economic capital and political capital of India \n {format_instructions}",
    input_variables= [],
    partial_variables= {"format_instructions": parser.get_format_instructions()}
)

chain= template | model | parser
response=chain.invoke({})
print(response)
print("Finance Capital:", response["finance_capital"])
print("Economic Capital:", response["economic_capital"])