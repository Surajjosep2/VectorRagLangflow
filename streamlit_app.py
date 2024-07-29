import streamlit as st
from langchain.vectorstores import AstraDBVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
import json

# Load the configuration from the JSON file
with open("Vector Store RAG_SJ.json") as f:
    config = json.load(f)

# Configuration data extracted from the JSON
api_endpoint = config["data"]["nodes"][1]["data"]["node"]["template"]["api_endpoint"]["value"]
collection_name = config["data"]["nodes"][1]["data"]["node"]["template"]["collection_name"]["value"]
token = config["data"]["nodes"][1]["data"]["node"]["template"]["token"]["value"]
namespace = config["data"]["nodes"][1]["data"]["node"]["template"]["namespace"]["value"]
embedding_model = config["data"]["nodes"][1]["data"]["node"]["template"]["embedding"]["name"]

# Initialize Vector Store and Embeddings
embeddings = OpenAIEmbeddings(model=embedding_model)
vector_store = AstraDBVectorStore(
    api_endpoint=api_endpoint,
    collection_name=collection_name,
    token=token,
    namespace=namespace,
    embeddings=embeddings,
)

# Create LLM instance
llm = OpenAI()

# Initialize Conversational Chain with Retrieval
retriever = vector_store.as_retriever()
prompt_template = PromptTemplate(input_variables=["context", "question"], template="{context}\n\nQuestion: {question}\n\nAnswer:")
memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")
qa = ConversationalRetrievalChain(
    retriever=retriever,
    llm=llm,
    memory=memory,
    prompt=prompt_template,
)

# Streamlit UI
st.title("RAG App with Astra DB and OpenAI")
st.write("This is a RAG application integrating Astra DB and OpenAI.")

# Input for user question
user_input = st.text_input("Ask a question:")

if user_input:
    response = qa({"question": user_input})
    st.write("Answer:", response["answer"])

    st.write("Chat History:")
    for chat in memory.chat_memory:
        st.write(f"User: {chat['user']}")
        st.write(f"AI: {chat['ai']}")
