from crewai import Agent, Task, Crew, Process
from groq import Groq
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd


## Reading the files to diagnose disease
try:
    reader = PdfReader('Skin_Diseases_and_Symptoms.pdf')
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    print(text)
except Exception as e:
    print(f"Error reading PDF: {e}")



# Language Model Inferencing
load_dotenv("credentials.txt")
groq_api_key = os.getenv("GROQ_API_KEY")
llama3 = ChatGroq(temperature=0, model_name="groq/llama-3.3-70b-versatile", api_key=groq_api_key)

def healthcare_agent(llm):
    return Agent(
    role='Dermatologist',
    goal='Define patient\'s skin disease with given symptoms.',
    backstory="You are Prof. at Dermatology and you can easily say the skin disease with given symptoms",
    llm=llama3, 
    verbose=True)

def healthcare_task(llm, user_input):
    return Task(
    description=f"Answer user's question {user_input} with using {text} this informations.",
    expected_output="Create a txt file and save the output in it. It must only contains the disease name!",
    agent=healthcare_agent(llm=llm),
    output_file="result.txt"
)

def main():
    st.title("DrSophia : A Dermatology Doctor 👩‍⚕️🩺")
    st.divider()
    st.write("Dr Sophia can diagnose up to 10 most common skin disease.")
    
    diseases = [
        "Acne",
        "Eczema",
        "Psoriasis",
        "Rosacea",
        "Hives (Urticaria)",
        "Dermatitis",
        "Skin Cancer",
        "Vitiligo",
        "Impetigo",
        "Shingles (Herpes Zoster)"
    ]
    
    diseases_df = pd.DataFrame({"Disease": diseases})
    
    st.header("Usage Hints:")
    st.write("-Write the symptoms of skin desease to get a diagnosis.")
    st.write("""DrSophia can diagnose these skin diseases:""")
    st.table(diseases_df)
    with st.sidebar:
        messages = st.container(height=400)
        if user_input:= st.chat_input("What are your symptoms?"):
            messages.chat_message("user").write(f"User has sent the following prompt: {user_input}")
            sophia_agent = healthcare_agent(llama3)
            sophia_task = healthcare_task(llama3, user_input=user_input)
            
            sophia_crew = Crew(agents=[sophia_agent],
                        tasks=[sophia_task], 
                        process=Process.sequential,
                        verbose=1,
                        max_rpm=29)
            output = sophia_crew.kickoff()
            messages.chat_message("assistant").write(output.raw)

if __name__ == "__main__":
    main()