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
st.markdown('''In this initial stage, the content from three specified websites is fetched \
and prepared for further processing. Using requests and BeautifulSoup libraries in Python, \
we retrieve the HTML content of each web page, then clean and parse the text to focus \
on the relevant information.
* Fetch content from each URL using requests.
* Parse the HTML with BeautifulSoup, extracting the main text content while removing any irrelevant HTML tags and navigation elements.
* Consolidate the content from all sources into a single text corpus, which will be processed in the next stages.''')

st.markdown(''':orange[2. Splitting and Chunking]''')
st.markdown('''Given the potential length of the extracted documents, \
splitting and chunking is crucial to ensure the data fits within \
the token limits of the language model and can be effectively retrieved. \
Using LangChain’s RecursiveCharacterTextSplitter, \
we break down the text into smaller, manageable chunks.
* Use the RecursiveCharacterTextSplitter class to split the text into chunks, \
setting a maximum chunk size (e.g., 500 tokens) with some overlap \
between chunks to preserve context.
* Store each chunk as a Document object with metadata indicating \
the source URL and chunk position. \
This allows for traceability and retrieval of the most relevant chunks later.''')

st.markdown(''':orange[3. Storage]''')
st.markdown('''The next step is to store these chunks as vector embeddings \
in a vector database, enabling efficient similarity search. \
We use OpenAI’s embedding model to convert each chunk into \
high-dimensional vector representations, \
then store these embeddings in a local vector storage FAISS \
accessible through LangChain.
* Initialize OpenAI embeddings using OpenAIEmbeddings.
* Generate embeddings for each text chunk.
* Store the embeddings in a vector store FAISS where \
each embedding is associated with the original document chunk \
and metadata. This setup enables fast retrieval of relevant content \
based on semantic similarity to a query.''')

st.markdown(''':orange[4. Retrieval]''')
st.markdown('''In this stage, when a user submits a query, \
the chatbot retrieves relevant document chunks from the vector store. \
The query is converted into an embedding and compared against the stored \
embeddings to find the top-matching chunks.
* Embed the user’s query using the same OpenAI embedding model.
* Search the vector store for document chunks that closely match \
the query embedding.
* Retrieve the top-k relevant chunks based on cosine similarity, \
which will be used as context for generating a response.''')

st.markdown(''':orange[5. Output]''')
st.markdown('''Finally, the retrieved chunks are used as context \
to generate a response through a language model. Using OpenAI’s 
gpt-4 model in LangChain’s RetrievalQA chain, \
the chatbot constructs a coherent answer based on the information \
in the retrieved chunks.
*Pass the retrieved chunks along with the user’s \
query as input to the language model using \
LangChain’s RetrievalQA chain.
* Generate a response that synthesizes information \
from the relevant chunks to directly address the query.
* Display the response to the user, completing the interaction''')

st.markdown('''For more domain-based content details and information, \
please visit [TP Special Educational Needs Support]\
(https://www.tp.edu.sg/life-at-tp/special-educational-needs-sen-support.html) and \
[Enabling Guide by SG Enable](https://www.enablingguide.sg/).''')

st.markdown(''':orange[This prototype is developed by Benedict Fernandez. \
Please feel free to give your comments, feedback and suggestions \
by reaching me at <bencmbit@gmail.com>. \
I would like to express my special thanks to Mr. Nick Tan from GovTech \
and my fellow learners at our pilot programme.]''')
