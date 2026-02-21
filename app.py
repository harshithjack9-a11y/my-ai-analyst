import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain.agents import create_pandas_dataframe_agent

st.title("AI Data Analyst")

file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if file:
    df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
    st.write("Data loaded:", df.shape)

    question = st.chat_input("Ask anything about the data")

    if question:
        llm = ChatGroq(model="llama-3.1-70b-versatile", groq_api_key=st.secrets["GROQ_API_KEY"])
        agent = create_pandas_dataframe_agent(llm, df, verbose=False, allow_dangerous_code=True)
        with st.spinner("Analyzing..."):
            answer = agent.run(question)
        st.write(answer)
