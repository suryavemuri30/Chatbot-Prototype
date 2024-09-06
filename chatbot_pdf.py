import streamlit as st
import os
from streamlit_chat import message
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as LangChainPinecone
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_community.llms import HuggingFaceHub

# Environment Variables
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ['PINECONE_API_KEY'] = "0ff9d5b7-1413-4646-a3d3-822ffc3fe8e9"

# Initialize Pinecone
api_key = os.environ['PINECONE_API_KEY']
pc = Pinecone(api_key=api_key)

# Define the index name
index_name1 = "langchain-chatbot2"

# Load documents and split them
directory = 'C:/Users/Surya Vemuri/Documents/test/content'

def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

documents = load_docs(directory)

def split_docs(documents, chunk_size=500, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs
    
docs = split_docs(documents)

# Use HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Index documents in Pinecone
vectorstore = PineconeVectorStore.from_documents(
    docs,
    index_name=index_name1,
    embedding=embeddings
)

# Set up the chatbot interface
st.title("Langchain Chatbot with Pinecone and Hugging Face")

# Initialize Hugging Face model
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",  # Using a larger model
    model_kwargs={
        "temperature": 0.7,  # Slightly increased for more diverse responses
        "max_length": 512,   # Increased to allow for longer responses
        "top_p": 0.95,       # Nucleus sampling for better text completion
    }
)

# Create ConversationalRetrievalChain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# Streamlit chat interface
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Hi! How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

query = st.text_input("Ask your question:")

if query:
    with st.spinner("Generating response..."):
        result = conversation_chain({"question": query})
        response = result['answer']

    st.session_state.requests.append(query)
    st.session_state.responses.append(response)

if st.session_state['responses']:
    for i in range(len(st.session_state['responses'])):
        message(st.session_state['responses'][i], key=str(i))
        if i < len(st.session_state['requests']):
            message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')