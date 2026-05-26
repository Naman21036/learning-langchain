from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature= 1.4
)

model= ChatHuggingFace(llm=llm)
parser= StrOutputParser()

prompt1= PromptTemplate(
    template= "You are a helpful assistant. Create a tweet about {topic}:",
    input_variables=["topic"]
)

prompt2= PromptTemplate(
    template= "You are a helpful assistant. Create a linkedin post about {topic}:",
    input_variables=["topic"]
)

chain= RunnableParallel({
    "tweet": RunnableSequence(prompt1, model, parser),
    "linkedin_post": RunnableSequence(prompt2, model, parser)
})

response= chain.invoke({"topic": "Artificial Intelligence"})
print(response)

chain.get_graph().print_ascii()