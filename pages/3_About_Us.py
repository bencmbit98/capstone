import streamlit as st

# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="ABC Capstone Project"
)
# endregion <--------- Streamlit App Configuration --------->

st.title("Supporting SEN Students")
st.write("in Temasek Polytechnic")

st.header("# About US")
st.title("Methodology")
st.write("This is a Streamlit App that demonstrates how to use the OpenAI API to generate text completions.")


st.write("**Important Notice**")
st.write("This web application is a prototype developed for educational purposes only. The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.")
st.write("Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.")
st.write("Always consult with qualified professionals for accurate and personalized advice.")
