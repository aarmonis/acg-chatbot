import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever


# PDF Processing Functions
def load_pdf(file_path):
    try:
        pdf_loader = UnstructuredPDFLoader(file_path=file_path)
        return pdf_loader.load()
    except ImportError as e:
        return f"ImportError: {str(e)}. Try installing the correct version of pdfminer.six."
    except Exception as e:
        return str(e)


def split_and_chunk(pdf_data, chunk_size=1000, chunk_overlap=100):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(pdf_data)
    except Exception as e:
        print(f"Error during splitting and chunking: {e}")
        return []


def create_vector_db(chunks, persist_directory="./chroma_db"):
    try:
        return Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(model="nomic-embed-text"),
            persist_directory=persist_directory,
        )
    except Exception as e:
        print(f"Error during creating vector database: {e}")
        return None


def process_pdf(file_path):
    pdf_data = load_pdf(file_path)
    if isinstance(pdf_data, str):  # Error occurred
        print(pdf_data)
        return None

    print(f"PDF loaded. Number of pages: {len(pdf_data)}")

    chunks = split_and_chunk(pdf_data)
    if not chunks:
        return None

    print(f"PDF split into {len(chunks)} chunks")

    vector_db = create_vector_db(chunks)
    if vector_db:
        print("Successfully created vector database")
        return vector_db
    else:
        print("Failed to create vector database")
        return None


# RAG System Setup
def setup_rag_system(vector_db):
    local_model = "gemma"
    llm = ChatOllama(model=local_model, device="cpu")

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )

    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# Main execution
def main():
    # Ensure Ollama is set up
    os.system("ollama list")

    # Process the PDF
    local_path = "data/OPAP_2022.pdf"
    vector_db = process_pdf(local_path)

    if vector_db is None:
        print("Failed to process PDF and create vector database. Exiting.")
        return

    # Set up RAG system
    rag_chain = setup_rag_system(vector_db)

    # Run interactive QA session
    while True:
        user_input = input("Enter your question (or 'quit' to exit): ")
        if user_input.lower() == "quit":
            break
        try:
            response = rag_chain.invoke(user_input)
            print("\nResponse:", response)
        except Exception as e:
            print(f"An error occurred: {e}")
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
