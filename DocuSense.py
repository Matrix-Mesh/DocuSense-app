from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import streamlit as st
from watsonxlangchain import LangChainInterface
import os
import tempfile

creds = {
    "apikey": "7pBaa5HpFLnaJZcgXt7pMWWkTNZKa74tIyDbmCrqIV08",
    "url": "https://us-south.ml.cloud.ibm.com",
}
llm = LangChainInterface(
    credentials=creds,
    model="meta-llama/llama-2-70b-chat",
    params={"decoding_method": "sample", "max_new_tokens": 200, "temperature": 0.5},
    project_id="427b077b-acee-4340-9596-bdd382804cdc",
)


@st.cache_resource
def load_pdf(temp_path):
    # Load the PDF from the temporary file path
    loaders = [PyPDFLoader(temp_path)]
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2"),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0),
    ).from_loaders(loaders)
    return index


st.title("Ask Watsonx 🤖")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    # Load the PDF and create the index
    index = load_pdf(temp_path)

    # Create the QA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=index.vectorstore.as_retriever(),
        input_key="question",
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Pass Your Prompt here")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = chain.run(prompt)

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Optionally, clean up the temporary file
    os.remove(temp_path)
else:
    st.info("Please upload a PDF file to get started.")
