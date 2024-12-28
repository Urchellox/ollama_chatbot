import streamlit as st
from llama_index.core.llms import ChatMessage
import logging
import time
from llama_index.llms.ollama import Ollama
import chromadb
from chromadb.config import Settings

logging.basicConfig(level=logging.INFO)

settings = Settings(
    persist_directory="chroma_data",
    allow_reset=True
)
client = chromadb.Client(settings)

def get_or_create_chroma_collection(name):
    try:
        return client.get_collection(name)
    except chromadb.errors.InvalidCollectionException:
        logging.info(f"Коллекция '{name}' не найдена. Создаём новую.")
        return client.create_collection(name)

chat_collection = get_or_create_chroma_collection("chat_history")

if 'messages' not in st.session_state:
    st.session_state.messages = []

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
    chat_collection.add(
        documents=[content],
        metadatas=[{"role": role}],
        ids=[f"{role}-{len(st.session_state.messages)}"],
    )

def retrieve_from_chroma():
    try:
        results = chat_collection.get()
        return results["documents"], results["metadatas"]
    except Exception as e:
        logging.error(f"Ошибка при извлечении данных: {e}")
        return [], []

def main():
    st.title("Chat with LLMs Models")
    logging.info("Приложение запущено")

    model = st.sidebar.selectbox("Выберите модель", ["llama3.2:latest", "llama3.1 8b", "phi3", "mistral"])
    logging.info(f"Выбрана модель: {model}")

    if prompt := st.chat_input("Ваш вопрос"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        store_in_chroma("user", prompt)
        logging.info(f"Ввод пользователя: {prompt}")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                start_time = time.time()
                logging.info("Генерация ответа")

                with st.spinner("Ответ пишется..."):
                    try:
                        messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in st.session_state.messages]
                        response_message = stream_chat(model, messages)
                        duration = time.time() - start_time
                        response_message_with_duration = f"{response_message}\n\nДлительность: {duration:.2f} секунд"
                        st.session_state.messages.append({"role": "assistant", "content": response_message_with_duration})
                        store_in_chroma("assistant", response_message_with_duration)
                        st.write(f"Длительность: {duration:.2f} секунд")
                        logging.info(f"Ответ: {response_message}, Длительность: {duration:.2f} секунд")

                    except Exception as e:
                        st.session_state.messages.append({"role": "assistant", "content": str(e)})
                        st.error("Произошла ошибка при генерации ответа.")
                        logging.error(f"Ошибка: {str(e)}")

    if st.sidebar.button("Показать сохранённую историю"):
        docs, metas = retrieve_from_chroma()
        st.sidebar.write("История сообщений:")
        for doc, meta in zip(docs, metas):
            st.sidebar.write(f"{meta['role']}: {doc}")

if __name__ == "__main__":
    main()
