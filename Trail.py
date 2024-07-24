from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.llms import Ollama
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os
import time
import shutil

# Directories setup
if not os.path.exists('files'):
    os.mkdir('files')

if not os.path.exists('data'):
    os.mkdir('data')

def clear_existing_data():
    if os.path.exists('data'):
        shutil.rmtree('data')
    os.mkdir('data')
    st.session_state.chat_history = []

def get_or_create_vectorstore(documents, pdf_name):
    if 'vectorstore' not in st.session_state or st.session_state.vectorstore is None or st.session_state.current_pdf != pdf_name:
        embedding_function = OllamaEmbeddings(model="mistral")
        vectorstore = FAISS.from_documents(documents, embedding_function)
        st.session_state.vectorstore = vectorstore
        st.session_state.current_pdf = pdf_name
    return st.session_state.vectorstore

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'current_pdf' not in st.session_state:
    st.session_state.current_pdf = None

if 'template' not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, here to help with questions about the information in the uploaded PDF. Your tone should be professional and informative. Always refer to the context provided to answer questions. If the information is not in the context, say so explicitly.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot: {response}"""

# Streamlit UI
st.title("PDF Question Answering System")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    try:
        file_path = os.path.join('files', uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Clear existing data
        clear_existing_data()

        # Load and split PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200))

        # Create vector store and add documents
        vectorstore = get_or_create_vectorstore(documents, uploaded_file.name)

        # Initialize language model for QA
        llm = Ollama(model="llama2")  # Using llama2 as an example, adjust as needed

        # Setup retriever and QA chain
        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        # Main QA Loop
        user_input = st.text_input("Ask your question:")
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Assistant is typing..."):
                    try:
                        response = qa_chain({"query": user_input})
                        answer = response['result']
                        source_docs = response['source_documents']

                        message_placeholder = st.empty()
                        full_response = ""
                        for chunk in answer.split():
                            full_response += chunk + " "
                            time.sleep(0.05)
                            message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)

                        # Display source documents
                        if source_docs:
                            st.write("Sources:")
                            for doc in source_docs:
                                st.write(f"- {doc.page_content[:200]}...")

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

                st.session_state.chat_history.append({"role": "assistant", "message": full_response})

    except Exception as e:
        st.error(f"An error occurred while processing the file: {str(e)}")

else:
    st.write("Please upload a PDF file.")