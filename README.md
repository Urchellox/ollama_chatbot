# ollama_chatbot
It-2302 Yuriy Mikhnevich

Chat with LLMs Models
This project allows users to chat with different LLM (Large Language Model) models through a web interface created using Streamlit. All messages are saved in a ChromaDB database for later use and analysis.

Features
Model Selection: The user can choose from available models such as Llama, Phi, or Mistral.
Chat with Model: The user can ask questions, and the model will generate responses in real-time.
Message History: All messages (from the user and the model) are saved in a ChromaDB database and can be viewed from the sidebar.
Response Generation Time: The time taken by the model to generate a response is displayed alongside the response.

Requirements
To run this project, you'll need to install the following dependencies:

streamlit — for creating the web interface
llama_index — for interacting with models
chromadb — for working with ChromaDB
ollama — for working with Ollama models

Install the necessary dependencies by running:
pip install -r requirements.txt

Running the Application
Make sure you have all the necessary dependencies installed.
Run the application with Streamlit:
streamlit run app.py

