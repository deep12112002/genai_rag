import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")  

groq_api_key = os.getenv('GROQ_API_KEY')

llm = ChatGroq(api_key=groq_api_key, model="llama3-8b-8192")  

prompt = ChatPromptTemplate.from_template(
    """
     Answer the questions based on the provided context only.  
     Please provide the most accurate response to the question.

    <context>
    {context}
    </context>

    Question: {input}

    Answer:
    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("research_paper")
        st.session_state.documents = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        st.session_state.final_document = st.session_state.text_splitter.split_documents(
            st.session_state.documents[:50]
        )
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_document,
            st.session_state.embeddings
        )

st.title("📚 RAG-Powered Document Q&A With Groq And Llama3")

user_input = st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.success("Vector database is ready!")

if user_input:
    with st.spinner("Generating answer..."):
        document_chain = create_stuff_documents_chain(llm, prompt)  ## Context of all the listed document
        retriever = st.session_state.vectors.as_retriever() ## quearing a database
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({'input': user_input})

        st.markdown("Answer:")
        st.write(response["answer"])

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('--------------------------')




    