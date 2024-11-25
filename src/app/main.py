import tkinter as tk
from tkinter import scrolledtext
from pymilvus import connections, Collection
import numpy as np
import requests
import json  

class Retriever:
    def __init__(self, host, port, collection_name, embedding_dim):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.connect_to_milvus()
        self.collection = Collection(self.collection_name)
        self.load_collection() 

    def connect_to_milvus(self):
        connections.connect("default", host=self.host, port=self.port)
        print("Connected to Milvus for retrieval.")

    def load_collection(self):
        print(f"Loading collection '{self.collection_name}' into memory.")
        self.collection.load()
        print(f"Collection '{self.collection_name}' loaded.")

        print(f"Number of entities in collection '{self.collection_name}': {self.collection.num_entities}")

    def get_embedding(self, text):
        try:
            response = requests.post(
                "http://127.0.0.1:11434/api/embed",  
                headers={"Content-Type": "application/json"},
                json={"model": "llama3.1:8b", "input": text}
            )
            print(f"Response Status Code: {response.status_code}")
            print("Response Headers:", response.headers)
            print(f"Response Text:\n{response.text}\n")

            response.raise_for_status()

            try:
                data = response.json()
            except ValueError as e:
                print(f"JSON decoding failed: {e}")
                return None

            if "embedding" in data and isinstance(data["embedding"], list):
                embedding = data["embedding"]
                print(f"Received embedding of length {len(embedding)}")
                return embedding
            elif "embeddings" in data and isinstance(data["embeddings"], list) and len(data["embeddings"]) > 0:
                embedding = data["embeddings"][0]
                print(f"Received embedding of length {len(embedding)}")
                return embedding
            else:
                print("Unexpected API response format.")
                print("API Response:", data)
                return None

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None

    def retrieve(self, query, top_k=5):
        embedding = self.get_embedding(query)
        if embedding is None:
            print("Failed to get embedding for the query.")
            return []

        print(f"Embedding for the query (first 5 values): {embedding[:5]}\n")

        search_params = {
            "metric_type": "IP",  
            "params": {"nprobe": 10},
        }
        try:
            results = self.collection.search(
                data=[embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["text"],
            )
            print("Search results obtained.")

            texts = [hit.entity.get("text") for hit in results[0]]
            print(f"Retrieved texts: {texts}\n")
            return texts
        except Exception as e:
            print(f"An error occurred during the search: {e}")
            return []

class Generator:
    def __init__(self, api_url):
        self.api_url = api_url

    def generate_response(self, prompt):
        try:
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json={"model": "llama3.1:8b", "prompt": prompt},
                stream=True  
            )
            print(f"Response Status Code: {response.status_code}")
            print("Response Headers:", response.headers)

            response.raise_for_status()

            full_response = ""

            for line in response.iter_lines(decode_unicode=True):
                if line:
                    print(f"Received line: {line}")
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            full_response += data["response"]
                        if data.get("done", False):
                            break  
                    except json.JSONDecodeError as e:
                        print(f"JSON decoding failed: {e}")
                        continue

            if full_response:
                print(f"Full response: {full_response}\n")
                return full_response
            else:
                print("Failed to generate a response from the API.")
                return "I'm sorry, I couldn't generate a response."

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return "I'm sorry, I couldn't generate a response."

class ChatbotUI:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator
        self.window = tk.Tk()
        self.window.title("RAG Chatbot")

        self.chat_display = scrolledtext.ScrolledText(self.window, wrap=tk.WORD, width=80, height=20)
        self.chat_display.pack(pady=10)

        self.user_input = tk.Entry(self.window, width=80)
        self.user_input.pack(pady=10)
        self.user_input.bind("<Return>", self.get_response)

        self.send_button = tk.Button(self.window, text="Send", command=self.get_response)
        self.send_button.pack()

    def get_response(self, event=None):
        user_query = self.user_input.get()
        self.chat_display.insert(tk.END, f"You: {user_query}\n")
        self.user_input.delete(0, tk.END)

        context_texts = self.retriever.retrieve(user_query)
        if not context_texts:
            response = "I'm sorry, I couldn't find any relevant information."
            self.chat_display.insert(tk.END, f"Bot: {response}\n")
            return

        context = "\n".join(context_texts)

        prompt = f"""You are an assistant that provides answers based on the following context.

Context:
{context}

Question: {user_query}
Answer:"""

        print(f"Generated prompt:\n{prompt}\n")

        response = self.generator.generate_response(prompt)
        self.chat_display.insert(tk.END, f"Bot: {response}\n")

    def run(self):
        self.window.mainloop()

def main():

    MILVUS_CONFIG = {
        "host": "127.0.0.1",
        "port": "19530",
        "collection_name": "embedded_texts",
        "embedding_dim": 4096, 
    }
    EMBEDDING_API_URL = "http://127.0.0.1:11434/api/embed"  
    GENERATION_API_URL = "http://127.0.0.1:11434/api/generate"  

    retriever = Retriever(
        MILVUS_CONFIG["host"],
        MILVUS_CONFIG["port"],
        MILVUS_CONFIG["collection_name"],
        MILVUS_CONFIG["embedding_dim"],
    )
    generator = Generator(GENERATION_API_URL)

    chatbot_ui = ChatbotUI(retriever, generator)
    chatbot_ui.run()

if __name__ == "__main__":
    main()
