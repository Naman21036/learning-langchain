from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough, RunnableBranch

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature= 1.4
)

model= ChatHuggingFace(llm=llm)
parser= StrOutputParser()

prompt1= PromptTemplate(
    template= "You are a helpful assistant. Write a detailed explanation about {topic}:",
    input_variables=["topic"]
)

prompt2= PromptTemplate(
    template= "You are a helpful assistant. Summarize the following text in 500 words: {text}",
    input_variables=["text"]
)

def count_words(text:str):
    return len(text.split())

chain1= RunnableSequence(prompt1, model, parser)
chain2= RunnableBranch(
    (lambda x: count_words(x["text"]) > 500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain= RunnableSequence(chain1, chain2)
#response= final_chain.invoke({"topic": "Politics"})
#print(response)
final_chain.get_graph().print_ascii()