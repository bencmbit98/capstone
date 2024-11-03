import streamlit as st

# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="ABC Capstone Project"
)
# endregion <--------- Streamlit App Configuration --------->

st.title("About this App")
st.markdown(''':green[At Temasek Polytechnic (TP), we are dedicated to fostering an inclusive\
and supportive learning environment for all students. \
Through our Special Educational Needs (SEN) Support Office, \
we offer tailored services that help students with diverse learning needs succeed academically\
and thrive personally.]''')

st.markdown(''':green[We provide individualized support plans, accessible learning resources,\
and career guidance to help SEN students overcome challenges and achieve their goals.\
We work closely with faculty and staff to create a campus culture that celebrates diversity\
and values each student’s unique contributions.]''')

st.markdown(''':orange[For more content details, please visit\
[TP Special Educational Needs Support](https://www.tp.edu.sg/life-at-tp/special-educational-needs-sen-support.html").]''')

st.markdown(''':orange[This prototype is developed by Benedict Fernandez. Please feel free to give your comments, feedback & suggestion. Reach me at <bencmbit@gmail.com>]''')

st.markdown('''**Important Notice**''')
st.markdown(''':red[This web application is a prototype developed for educational purposes only. \
The information provided here is NOT intended for real-world usage and \
should not be relied upon for making any decisions, 
especially those related to financial, legal, or healthcare matters.]''')
st.markdown(''':red[Furthermore, please be aware that the LLM may generate inaccurate or \
incorrect information. You assume full responsibility for how you use any generated output.]''')
st.markdown(''':red[Always consult with qualified professionals for accurate and personalized advice.]''')
