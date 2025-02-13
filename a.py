import streamlit as st
from llama_index.core.llms import ChatMessage
import logging
import time
from llama_index.llms.ollama import Ollama
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

logging.basicConfig(level=logging.INFO)

settings = Settings(
    persist_directory="chroma_data",
    allow_reset=True
)
client = chromadb.Client(settings)

encoder = SentenceTransformer("all-MiniLM-L6-v2")

def get_or_create_chroma_collection(name):
    try:
        return client.get_collection(name)
    except chromadb.errors.InvalidCollectionException:
        logging.info(f"Collection '{name}' not found. Creating a new one.")
        return client.create_collection(name)

chat_collection = get_or_create_chroma_collection("chat_history")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_contents" not in st.session_state:
    st.session_state.file_contents = []

def stream_chat(model, messages):
    try:
        llm = Ollama(model=model, request_timeout=120.0)
        resp = llm.stream_chat(messages)
        response = ""
        response_placeholder = st.empty()
        for r in resp:
            response += r.delta
            response_placeholder.write(response)
        logging.info(f"Model: {model}, Messages: {messages}, Response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error during streaming: {str(e)}")
        raise e

def store_in_chroma(role, content):
    vector = encoder.encode(content).tolist()
    chat_collection.add(
        documents=[content],
        metadatas=[{"role": role}],
        ids=[f"{role}-{len(st.session_state.messages)}"],
        embeddings=[vector]
    )

def retrieve_from_chroma(query, top_k=3):
    try:
        query_vector = encoder.encode(query).tolist()
        results = chat_collection.query(query_embeddings=[query_vector], n_results=top_k)
        return results["documents"][0] if results["documents"] else []
    except Exception as e:
        logging.error(f"Error retrieving data: {e}")
        return []

def generate_multi_queries(query):
    variations = [
        f"Rephrase: {query}",
        f"Explain differently: {query}",
        f"Clarify in another way: {query}"
    ]
    return variations

def load_constitution(file_path="constitution_kz.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        logging.info("Constitution text loaded successfully.")
        store_in_chroma("system", content)
    except FileNotFoundError:
        logging.error("Constitution file not found. Ensure the file exists.")

def rag_fusion(queries, top_k=3):
    all_results = []
    for query in queries:
        retrieved_docs = retrieve_from_chroma(query, top_k)
        all_results.extend(retrieved_docs)

    if not all_results:
        return []

    unique_docs = list(set(all_results))
    doc_embeddings = np.array([encoder.encode(doc).tolist() for doc in unique_docs])

    query_embeddings = np.array([encoder.encode(q).tolist() for q in queries])
    mean_query_embedding = np.mean(query_embeddings, axis=0)

    similarities = np.dot(doc_embeddings, mean_query_embedding) / (np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(mean_query_embedding))
    ranked_docs = [doc for _, doc in sorted(zip(similarities, unique_docs), reverse=True)]

    return ranked_docs[:5]

def main():
    st.title("Chat with LLM Models")
    logging.info("Application started")

    model = st.sidebar.selectbox("Select Model", ["llama3.2:latest", "llama3.1 8b", "phi3", "mistral"])
    logging.info(f"Selected model: {model}")
    
    load_constitution()

    uploaded_files = st.sidebar.file_uploader("Upload text files", type="txt", accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_content = uploaded_file.read().decode("utf-8")
            st.session_state.file_contents.append(file_content)
            store_in_chroma("system", file_content)
        st.sidebar.success("Files successfully uploaded and added to context.")

    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        store_in_chroma("user", prompt)
        logging.info(f"User input: {prompt}")

        queries = generate_multi_queries(prompt)
        relevant_contexts = rag_fusion(queries, top_k=3)
        
        relevant_contexts.extend(st.session_state.file_contents)
        unique_contexts = list(set(relevant_contexts))[:5]
        context_text = "\n".join(unique_contexts)
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                start_time = time.time()
                logging.info("Generating response")

                with st.spinner("Writing response..."):
                    try:
                        messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in st.session_state.messages]
                        messages.insert(0, ChatMessage(role="system", content=context_text))
                        response_message = stream_chat(model, messages)
                        duration = time.time() - start_time
                        response_message_with_duration = f"{response_message}\n\nResponse Time: {duration:.2f} seconds"
                        st.session_state.messages.append({"role": "assistant", "content": response_message_with_duration})
                        store_in_chroma("assistant", response_message_with_duration)
                        st.write(f"Response Time: {duration:.2f} seconds")
                        logging.info(f"Response: {response_message}, Duration: {duration:.2f} seconds")
                    except Exception as e:
                        st.session_state.messages.append({"role": "assistant", "content": str(e)})
                        st.error("An error occurred while generating the response.")
                        logging.error(f"Error: {str(e)}")

    if st.sidebar.button("Show Saved History"):
        all_messages = chat_collection.get(include=["documents", "metadatas"])
        if all_messages and "documents" in all_messages:
            st.sidebar.write("Chat History:")
            for i, doc in enumerate(all_messages["documents"]):
                role = all_messages["metadatas"][i]["role"]
                st.sidebar.write(f"**{role.capitalize()}**: {doc}")

if __name__ == "__main__":
    main()
