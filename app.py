import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

st.title("AI Data Analyst")

file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if file is not None:
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        st.success(f"Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(8))

        question = st.chat_input("Ask anything about your data... (e.g. 'What is the average of column X?' or 'Plot sales trend')")

        if question:
            # Use current working Groq model (Feb 2026)
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                groq_api_key=st.secrets["GROQ_API_KEY"],
                temperature=0.2
            )

            # Create pandas agent
            agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=False,
                allow_dangerous_code=True,
                handle_parsing_errors=True
            )

            with st.spinner("Analyzing..."):
                try:
                    answer = agent.run(question)
                    st.markdown("**Answer:**")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
                    st.info("Try a simpler question, check data format, or ensure your Groq key is valid.")

    except Exception as e:
        st.error(f"Could not read the file: {str(e)}")
        st.info("Supported: clean CSV or XLSX. Make sure openpyxl and tabulate are in requirements.txt.")
