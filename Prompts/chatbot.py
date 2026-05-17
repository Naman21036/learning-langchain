from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()
llm= HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    temperature= 1.4
)

chat_history= [SystemMessage(content="You are a helpful assistant.")]
while True:
    user_input= input("User:")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chatbot. Goodbye!")
        break
    chat_history.append(HumanMessage(content=user_input))
    response= ChatHuggingFace(llm=llm).invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("LLM Response:", response.content)
print("Chat history:", chat_history)