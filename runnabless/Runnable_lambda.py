from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature= 1.4
)

model= ChatHuggingFace(llm=llm)
parser= StrOutputParser()

prompt= PromptTemplate(
    template= "You are a helpful assistant. Write a joke about {topic}:",
    input_variables=["topic"]
)

def count_words(text:str):
    return len(text.split())

chain1= RunnableSequence(prompt, model, parser)
chain2= RunnableParallel({
    "word_count": RunnableLambda(count_words),
    "passthrough": RunnablePassthrough()
})

final_chain= RunnableSequence(chain1, chain2)
#response= final_chain.invoke({"topic": "Politics"})
#print(response)
final_chain.get_graph().print_ascii()