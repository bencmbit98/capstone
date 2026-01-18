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
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.docstore.document import Document # Import Document class
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from helper_functions import llm
# from logics.customer_query_handler import process_user_message
# from helper_functions.utility import check_password

# Helper Functions =============================================
# This is the helper function for calling LLM
def get_completion(prompt,
                   model="gpt-4o-mini",
                   temperature=0,
                   top_p=1.0,
                   max_tokens=256,
                   n=1,
                   json_output=False):

    # To check if there is an output in json object format
    if json_output == True:
      output_json_structure = {"type": "json_object"}
    else:
      output_json_structure = None

    # define messages
    messages = [{"role": "user", "content": prompt}]

    # Get LLM response
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=1,
        response_format=output_json_structure,
    )
    # print(response)
    return response.choices[0].message.content

# This function directly take in "messages" as the parameter.
def get_completion_by_messages(messages, model="gpt-4o-mini", temperature=0, top_p=1.0, max_tokens=1024, n=1):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=1
    )
    return response.choices[0].message.content

# Functions for Counting Tokens
def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    return len(encoding.encode(text))

def count_tokens_from_message(messages):
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    value = ' '.join([x.get('content') for x in messages])
    return len(encoding.encode(value))

# Function for Generating Embedding
def get_embedding(input, model='text-embedding-3-small'):
    response = client.embeddings.create(
        input=input,
        model=model
    )
    return [x.embedding for x in response.data]

# RAG Step 1: Document Loading ============================================================
def RAG_Load():
    webpage_urls = [
        'https://www.tp.edu.sg/life-at-tp/special-educational-needs-sen-support.html',
        'https://www.enablingguide.sg/im-looking-for-disability-support/transport',
        'https://www.enablingguide.sg/im-looking-for-disability-support/child-adult-care',
        'https://www.enablingguide.sg/im-looking-for-disability-support/training-employment']
    
    all_documents = []
    
    for url in webpage_urls:
        response = requests.get(url)
        soup = bs4.BeautifulSoup(response.content, 'html.parser')
        text = soup.text.replace('\n', ' ')
        # Wrap the web text in a Document object to match PDF format
        all_documents.append(Document(page_content=text, metadata={"source": url}))
    
    # --- Part B: Load from PDFs ---
    pdf_loader = PyPDFDirectoryLoader("data/")
    pdf_docs = pdf_loader.load()
    
    # Combine both lists
    all_documents.extend(pdf_docs)
    
    return all_documents

# RAG Step 2: Splitting and Chunking
def RAG_SplittingChunking(all_documents):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=500,
        chunk_overlap=50,
        length_function=count_tokens
    )

    # Correct Action: Pass the list of documents directly to the splitter.
    # We remove the "document = Document(...)" line because all_documents 
    # already contains the processed data from your RAG_Load function.
    splitted_documents = text_splitter.split_documents(all_documents)

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

st.title("Support SEN Student")
st.write("in Temasek Polytechnic")
# Check if the password is correct.  
if not check_password():  
    st.stop()

final_text = RAG_Load()
splitted_documents = RAG_SplittingChunking(final_text)
qa_chain = RAG_Storage(splitted_documents)

form = st.form(key="form")
form.subheader("Prompt")

user_prompt = form.text_area("Ask me anything: ", height=200)

if form.form_submit_button("Submit"):
    
    st.toast(f"You asked - {user_prompt}")

    st.divider()   
    response = qa_chain.invoke(user_prompt)
    st.write(response)
    st.divider()
