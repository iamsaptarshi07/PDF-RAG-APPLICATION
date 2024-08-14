import streamlit as st
import pymupdf
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import google.generativeai as genai
from dotenv import load_dotenv


def get_pdf_text(uploaded_files) -> str:
    text = ""

    # Check if the input is a directory
    if os.path.isdir(uploaded_files):
        # Iterate over all PDF files in the directory
        for filename in os.listdir(uploaded_files):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(uploaded_files, filename)
                with pymupdf.open(pdf_path) as doc:
                    for page in doc:
                        text += page.get_text()
    elif os.path.isfile(uploaded_files):
        if os.path.isfile(uploaded_files):
                # Handle a single PDF
            with pymupdf.open(uploaded_files) as doc:
                for page in doc:
                    text += page.get_text()
    else:
        raise ValueError(f"The path {uploaded_files} is neither a file nor a directory.")

    return text


def process_pdf(text: str) -> Chroma:
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    documents = [Document(page_content=chunk) for chunk in chunks]
    vectors = Chroma(embedding_function=embeddings, persist_directory="./db")
    vectors.add_documents(documents=documents)
    return vectors


def instantiate_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash-001")


def generate_response(question: str, retriever: MultiQueryRetriever):
    template = """Answer the questions based ONLY on the following context:
    {context}
    Question: {question}
    """
    llm = instantiate_llm()
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response = chain.invoke(input=question)
    return response

def instantiateAPI() -> None:
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def main():
    st.title("PDF RAG Application")

    # File uploader for PDFs
    uploaded_files = st.file_uploader("Choose a PDF file", type="pdf")

    instantiateAPI()

    if uploaded_files is not None:
        # Save uploaded file to a temporary location
        if not os.path.exists("temp"):
            os.mkdir("temp")
        pdf_path = os.path.join("temp", uploaded_files.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_files.getbuffer())

        # Process the PDF and create a retriever
        st.write("Processing PDF...")
        raw_text = get_pdf_text(uploaded_files=pdf_path)
        vectors = process_pdf(raw_text)

        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five
            different versions of the given user question to retrieve relevant documents from a vector
            database. By generating multiple perspectives on the user question, your goal is to help
            the user overcome some of the limitations of the distance-based similarity search.
            Provide these alternative questions separated by newlines.
            Original question: {question}""",
        )

        retriever = MultiQueryRetriever.from_llm(
            vectors.as_retriever(search_kwargs={"k": 5}),
            instantiate_llm(),
            prompt=QUERY_PROMPT,
        )

        st.write("PDF processed successfully.")

        # Text input for user queries
        question = st.text_input("Ask a question about the PDF:")

        if st.button("Submit"):
            st.write("Generating response...")
            response = generate_response(question, retriever)
            st.write("Response:")
            st.write(response)


if __name__ == "__main__":
    main()
