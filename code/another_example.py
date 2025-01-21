from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os
import time


from langchain_community.llms import Ollama




template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"""
prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )

vectorstore = Chroma(persist_directory='chromadb',
                    embedding_function=OllamaEmbeddings(base_url='http://localhost:11434',
                                                                              model="llama3.1"))

llm = Ollama(base_url="http://localhost:11434",
                                  model="llama3.1",
                                  verbose=True
                                  )


loader = PyPDFLoader("../data/pdfs/Making_biodiversity_work_for_coffee_prod.pdf")
data = loader.load()

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    length_function=len
)
all_splits = text_splitter.split_documents(data)
print('splitted')
# Create and persist the vector store
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=OllamaEmbeddings(model="llama3.1")
)
vectorstore.persist()
print('embeddings')
retriever = vectorstore.as_retriever()
# Initialize the QA chain
qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
        }
    )


# Generate the answer to a question
def answer_question(question):
    try:
        response = qa_chain.run(question)
        print(f"User: {question}")
        print(f"Chatbot: {response}")
        return response
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
if __name__ == "__main__":
    user_question = "What are the key benefits of biodiversity for coffee production?"
    answer = answer_question(user_question)