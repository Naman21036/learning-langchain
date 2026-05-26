from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

with open("chains/template.txt", "r") as f:
    text = f.read()

llm1= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature= 1.4
)

llm2= HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    temperature= 1.4
)

model1= ChatHuggingFace(llm=llm1)
model2= ChatHuggingFace(llm=llm2)

parser= StrOutputParser()
prompt1= PromptTemplate(
    template="You are a helpful assistant. Please make notes on: {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="You are a helpful assistant. Please generate exam focused questions on : {text}\n\n",
    input_variables=["text"]
)

prompt3= PromptTemplate(
    template= "You are a helpful assistant. Please merge the following notes -> {notes} and questions -> {questions} into a single document",
    input_variables=["notes", "questions"]
)

parallel_chain= RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "questions": prompt2 | model2 | parser
})

merge_chain= prompt3 | model2 | parser

final_chain= parallel_chain | merge_chain

response= final_chain.invoke({"text": text})
print(response)

