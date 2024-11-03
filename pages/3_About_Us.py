import streamlit as st

# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="ABC Capstone Project"
)
# endregion <--------- Streamlit App Configuration --------->

st.title("About this App")
st.write("At Temasek Polytechnic (TP), we are committed to providing an inclusive, 
supportive, and empowering environment for all students. 
We believe that every student should have equal opportunities to succeed, grow, 
and thrive in their educational journey. 
Our Special Educational Needs (SEN) Support Office is dedicated to assisting students 
with diverse learning needs, ensuring they feel welcome, respected, and supported throughout their time at TP.")


st.write("**Important Notice**")
st.write("This web application is a prototype developed for educational purposes only. The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.")
st.write("Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.")
st.write("Always consult with qualified professionals for accurate and personalized advice.")
