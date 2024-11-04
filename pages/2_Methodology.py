import streamlit as st

st.header("Methodology for RAG Chatbot Development")
st.write("Prototype by Benedict Fernandez (TP)")

st.markdown(''':green[This chatbot prototype uses a **naive Retrieval-Augmented Generation (RAG) model** \
to generate responses based on content from three external websites. \
By combining retrieval and generation, the model provides accurate, \
contextually relevant answers to user queries. \
Below is the detailed methodology for the chatbot’s development, structured in five sequential stages: \
Document Loading, Splitting & Chunking, Storage, Retrieval, and Output.]''')

st.markdown(''':orange[1. Document Loading]''')
st.markdown('''\
In this initial stage, the content from three specified websites is fetched \
and prepared for further processing. Using requests and BeautifulSoup libraries in Python, \
we retrieve the HTML content of each web page, then clean and parse the text to focus \
on the relevant information.
* Fetch content from each URL using requests.
* Parse the HTML with BeautifulSoup, extracting the main text content while removing any irrelevant HTML tags and navigation elements.
* Consolidate the content from all sources into a single text corpus, which will be processed in the next stages.
\''')

st.markdown(''':orange[2. Splitting & Chunking]''')
st.markdown(''':green[body]''')

st.markdown(''':orange[3. Storage]''')
st.markdown(''':green[body]''')

st.markdown(''':orange[4. Retrieval]''')
st.markdown(''':green[body]''')

st.markdown(''':orange[5. Output]''')
st.markdown(''':green[body]''')

st.markdown(''':orange[For more content details and information, please visit [TP Special Educational Needs Support](https://www.tp.edu.sg/life-at-tp/special-educational-needs-sen-support.html) and [Enabling Guide by SG Enable](https://www.enablingguide.sg/).]''')

st.markdown(''':orange[This prototype is developed by Benedict Fernandez. Please feel free to give your comments, feedback & suggestions by reaching me at <bencmbit@gmail.com>. I would like to give my special thanks to Mr. Nick Tan from GovTech and my fellow learners at our pilot programme.]''')
