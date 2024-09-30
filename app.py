# Import necessary libraries
import os
# from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load the environment variables (commented out for now)
# load_dotenv()
os.environ['GROQ_API_KEY'] = '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'

# Get the working directory of the current script
working_dir = os.path.dirname(os.path.abspath(__file__))

# Function to load documents from a PDF file
def load_document(file_path):
    loader = UnstructuredPDFLoader(file_path)
    documents = loader.load()
    return documents

# Function to set up the vector store with document embeddings
def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings()
    # embeddings = OllamaEmbeddings(model="llama-3.1-70b-versatile",)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    doc_chunks = text_splitter.split_documents(documents)
    # Initialize the vector store with document chunks and embeddings
    vectorstore = Chroma.from_documents(documents=doc_chunks, embedding=embeddings)
    return vectorstore

# Function to create a conversational retrieval chain
def create_chain(vectorstore):
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0.7
    )
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    # Create the conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    return chain

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Chat with Doc",
    page_icon="📄",
    layout="centered"
)

# Display the title of the Streamlit app
st.title("🦜🔗 LangChain Doc Chat - Experience LLAMA 3.1")

# Initialize the chat history in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File uploader widget to upload a PDF file
uploaded_file = st.file_uploader(label="Upload your pdf file", type=["pdf"])

if uploaded_file:
    file_path = f"{working_dir}/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Set up the vector store if not already done
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = setup_vectorstore(load_document(file_path))

    # Create the conversational chain if not already done
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

# Display the chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input widget for user to ask questions
user_input = st.chat_input("Ask Llama...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Display the user's message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get the assistant's response and display it
    with st.chat_message("assistant"):
        response = st.session_state.conversation_chain({"question": user_input})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
