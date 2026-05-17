from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt 
load_dotenv()
template= load_prompt("template.json")
llm= HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    temperature= 1.2
)

st.header("PromptGPT")
st.write("A simple LLM app using HuggingFace and Streamlit")
st.write("Ask any question to the LLM and get an answer")
company= st.text_input("Enter your company")
role= st.text_input("Enter the role")
skills= st.text_input("Enter the skills")
experience= st.text_input("Enter the experience")
prompt= template.invoke({"company": company, "role": role, "skills": skills, "experience": experience})
if st.button("Get Response"):
    response= ChatHuggingFace(llm=llm).invoke(prompt)
    st.write(response.content)