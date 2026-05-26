from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    temperature= 1.4
)

model= ChatHuggingFace(llm=llm)
parser= StrOutputParser()
prompt1= PromptTemplate(
    template="You are a helpful assistant. Please make a detailed report on: {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="You are a helpful assistant. Please generate 5 key points of: {text}\n\n",
    input_variables=["text"]
)

chain= prompt1 | model | parser | prompt2 | model | parser 
response=chain.invoke({"topic": "Chess"})
print(response)

##from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    temperature= 1.4
)

model= ChatHuggingFace(llm=llm)
parser= StrOutputParser()
prompt1= PromptTemplate(
    template="You are a helpful assistant. Please answer the following question in 10 lines: {question}"
)

chain= prompt1 | model | parser
response=chain.invoke({"question": "What are the key aspects of polygon's technology stack?"})
print(response)

chain.get_graph().print_ascii()