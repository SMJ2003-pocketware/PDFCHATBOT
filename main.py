import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    # Keep smaller chunks for better context retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Use allow_dangerous_deserialization=True when loading FAISS index
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Retrieve more document chunks for a larger context
    docs = new_db.similarity_search(user_question, k=5)  # Increased to 5 chunks
    
    # Concatenate the retrieved chunks to form a larger context for the QA model
    context = " ".join([doc.page_content for doc in docs])
    
    if context.strip():  # Proceed only if there's valid context
        # Using a more generative model like flan-t5-base for detailed answers
        qa_model = pipeline("text2text-generation", model="google/flan-t5-base")
        
        prompt = f"Question: {user_question}\nContext: {context}\nProvide a detailed answer:"
        
        response = qa_model(prompt, max_length=512)  # Increase max_length for longer answers
        
        st.write("Reply: ", response[0]['generated_text'])
    else:
        st.write("No relevant context found for your question.")

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Hugging Face:")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        with st.spinner("Retrieving answer..."):
            user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()