from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
#Now we use pydantic parsers with structured output instead of structured output parser
load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    temperature= 1.4
)

model= ChatHuggingFace(llm=llm)
schema= [
    ResponseSchema(name="fact_1", description=" it is the fact 1 about the topic"),
    ResponseSchema(name="fact_2", description=" it is the fact 2 about the topic"),
    ResponseSchema(name="fact_3", description=" it is the fact 3 about the topic")
]
parser= StructuredOutputParser(response_schemas=schema)

template= PromptTemplate(
    template= "Give me 3 facts about {topic}. {format_instructions}",
    input_variables= ["topic"],
    partial_variables= {"format_instructions": parser.get_format_instructions()}
)

prompt= template.format(topic="India")
response= model.invoke(prompt)
parsed_response= parser.parse(response.content)
print(parsed_response) 