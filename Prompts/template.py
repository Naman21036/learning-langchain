from langchain_core.prompts import PromptTemplate   
template= PromptTemplate(
    input_variables=["company", "role", "skills", "experience"],
    template="""
You are an AI hiring assistant.

Analyze the following candidate profile.

Company: {company}
Role: {role}
Skills: {skills}
Experience: {experience}

Provide:
1. Candidate fit score out of 10
2. Strengths of the candidate
3. Missing skills
4. Suggested interview questions
5. Final hiring recommendation
"""
)
template.save("template.json")