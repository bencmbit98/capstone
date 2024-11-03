# Set up and run this Streamlit App
import streamlit as st
import pandas as pd
import tiktoken
import requests
import openai
import bs4
from bs4 import BeautifulSoup
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from helper_functions import llm
from helper_functions.utility import check_password



# RAG Step 1: Document Loading ============================================================
def RAG_Load():
    webpage_urls = [
        'https://www.tp.edu.sg/life-at-tp/special-educational-needs-sen-support.html',
        'https://www.enablingguide.sg/im-looking-for-disability-support/transport',
        'https://www.enablingguide.sg/im-looking-for-disability-support/child-adult-care',
        'https://www.enablingguide.sg/im-looking-for-disability-support/training-employment']
    
    final_text = ""
    for url in webpage_urls:
        response = requests.get(url)
        soup = bs4.BeautifulSoup(response.content, 'html.parser')
        final_text += soup.text.replace('\n', '')
    
    return final_text

# RAG Step 2: Splitting and Chunking
def RAG_SplittingChunking(final_text):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.docstore.document import Document # Import Document class

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=500,
        chunk_overlap=50,
        length_function=count_tokens
    )

    # Create a Document object with the text content
    document = Document(page_content=final_text)

    # Now pass the Document object to split_documents
    splitted_documents = text_splitter.split_documents([document]) # Pass a list containing the Document object
  
    # Show the number of tokens in each of the splitted documents
    # for doc in splitted_documents:
    #    print(count_tokens(doc.page_content))

    return splitted_documents


# RAG Step 3: Storage
def RAG_Storage(splitted_documents):
    # An embeddings model is initialized using the OpenAIEmbeddings class.
    embeddings_model = OpenAIEmbeddings(model='text-embedding-ada-002')

    # Use FAISS for vector storage
    vector_store = FAISS.from_documents(splitted_documents, embeddings_model)
    
    # Initialize RetrievalQA with FAISS as retriever
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model='gpt-4o-mini'),
        retriever=vector_store.as_retriever(k=3)
    )
    
    return qa_chain # Return the qa_chain object
    
    # Show the number of documents in the vector store
    # vector_store._collection.count()

    # Peek at one of the documents in the vector store
    # vector_store._collection.peek(limit=1)

# RAG Step 4: Retrieval
# vector_store.similarity_search('taxi', k=3)
# vector_store.similarity_search_with_relevance_scores('taxi', k=3)

# RAG Step 5: Output
# result = qa_chain.invoke("How can get a taxi?")

# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="ABC Capstone Project"
)
# endregion <--------- Streamlit App Configuration --------->

st.title("Supporting SEN Students")
st.write("in Temasek Polytechnic")

# Main RAG Process Flow
final_text = RAG_Load()
splitted_documents = RAG_SplittingChunking(final_text)
qa_chain = RAG_Storage(splitted_documents)

# Interface
form = st.form(key="form")
form.subheader("Ask Me Anything")

user_prompt = form.text_area("Related to Special Educational Needs Support: ", height=50)

if form.form_submit_button("Send"):
    st.toast(f"Please wait while I seek answers to your query '{user_prompt}'")   
    response = qa_chain.invoke(user_prompt)
    answer = response["result"]
    st.write(answer)
    if st.button("OK! I am happy with your answers!"):
      st.balloons()
    with st.expander("Important Disclaimer"):
      st.write("IMPORTANT NOTICE: This web application is a prototype developed for educational purposes only. The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.")
      st.write("Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.")
      st.write("Always consult with qualified professionals for accurate and personalized advice.")
