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

prompt1= PromptTemplate(
    template="You are a helpful assistant. Please answer the following question in 10 lines: {question}"
)

prompt2 = PromptTemplate(
    template="You are a helpful assistant. Please generate summary of: {text}\n\nAnswer:"
)

parser= StrOutputParser()

chain= prompt1 | model | parser | prompt2 | model | parser

response=chain.invoke({"question": "What are the key aspects of polygon's technology stack?"})
print(response)