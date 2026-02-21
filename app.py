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

        question = st.chat_input("Ask anything about your data...")

        if question:
            # Initialize LLM
            llm = ChatGroq(
                model="llama-3.1-70b-versatile",
                groq_api_key=st.secrets["GROQ_API_KEY"],
                temperature=0.2
            )

            # Create agent
            agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=False,
                allow_dangerous_code=True,          # required in recent versions
                handle_parsing_errors=True
            )

            with st.spinner("Analyzing..."):
                try:
                    answer = agent.run(question)
                    st.markdown("**Answer:**")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.info("Try a simpler question or check your data format.")

    except Exception as e:
        st.error(f"Could not read the file: {str(e)}")
