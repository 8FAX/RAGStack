import tkinter as tk
from tkinter import scrolledtext, messagebox
from tkinter import ttk
import socket
import sqlite3
import requests
import json
import datetime
from pymilvus import connections, Collection
import uuid

# -----------------------------
# Database setup and utilities
# -----------------------------

USER_DB_PATH = "user.db"
CHATS_DB_PATH = "chats.db"

def create_user_table():
    conn = sqlite3.connect(USER_DB_PATH)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS user(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        uuid TEXT,
        created_at TEXT
    )
    ''')
    conn.commit()
    conn.close()

def create_chats_tables():
    conn = sqlite3.connect(CHATS_DB_PATH)
    c = conn.cursor()

    # Table for chats
    c.execute('''
    CREATE TABLE IF NOT EXISTS chats(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_uuid TEXT,
        title TEXT,
        message_count INTEGER DEFAULT 0
    )
    ''')

    # Table for messages
    c.execute('''
    CREATE TABLE IF NOT EXISTS messages(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER,
        sender TEXT,
        pos INTEGER,
        message TEXT,
        FOREIGN KEY(chat_id) REFERENCES chats(id)
    )
    ''')

    conn.commit()
    conn.close()

def insert_user(username, user_uuid):
    conn = sqlite3.connect(USER_DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO user (username, uuid, created_at) VALUES (?,?,?)", (username, user_uuid, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def get_user(id: str):
    conn = sqlite3.connect(USER_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT username, uuid FROM user WHERE username=?", (id,))
    print(f"id: {id}")
    row = c.fetchone()
    print(row[1])
    conn.close()
    return row[1] if row else None

def insert_chat(user_uuid, title=None):
    conn = sqlite3.connect(CHATS_DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO chats (user_uuid, title, message_count) VALUES (?,?,0)", (user_uuid, title))
    chat_id = c.lastrowid
    conn.commit()
    conn.close()
    return chat_id

def update_chat_title(chat_id, title):
    conn = sqlite3.connect(CHATS_DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE chats SET title=? WHERE id=?", (title, chat_id))
    conn.commit()
    conn.close()

def get_all_chats(user_uuid):
    conn = sqlite3.connect(CHATS_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, title FROM chats WHERE user_uuid=? ORDER BY id DESC", (user_uuid,))
    rows = c.fetchall()
    conn.close()
    return rows

def get_chat_messages(chat_id):
    conn = sqlite3.connect(CHATS_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT sender, pos, message FROM messages WHERE chat_id=? ORDER BY pos ASC", (chat_id,))
    rows = c.fetchall()
    conn.close()
    return rows

def insert_message(chat_id, sender, pos, message):
    conn = sqlite3.connect(CHATS_DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO messages (chat_id, sender, pos, message) VALUES (?,?,?,?)", (chat_id, sender, pos, message))
    c.execute("UPDATE chats SET message_count=message_count+1 WHERE id=?", (chat_id,))
    conn.commit()
    conn.close()

def get_chat_message_count(chat_id):
    conn = sqlite3.connect(CHATS_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT message_count FROM chats WHERE id=?", (chat_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else 0

# -----------------------------
# Auth and Network Utilities
# -----------------------------
def send_data(data):
    HOST = '127.0.0.1'  # your auth server host
    PORT = 9999         # your auth server port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(data.encode('utf-8'))
        response = s.recv(1024).decode('utf-8')
    return response

# -----------------------------
# Retriever and Generator
# -----------------------------
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
        print(f"Number of entities: {self.collection.num_entities}")

    def get_embedding(self, text):
        try:
            response = requests.post(
                "http://127.0.0.1:11434/api/embed",
                headers={"Content-Type": "application/json"},
                json={"model": "snowflake-arctic-embed2:latest", "input": text}
            )
            print(f"Response Status Code: {response.status_code}")
            response.raise_for_status()
            data = response.json()
            # Handling embedding
            if "embedding" in data and isinstance(data["embedding"], list):
                return data["embedding"]
            elif "embeddings" in data and isinstance(data["embeddings"], list) and len(data["embeddings"]) > 0:
                return data["embeddings"][0]
            else:
                print("Unexpected embedding response format.")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None

    def retrieve(self, query, top_k=25):
        embedding = self.get_embedding(query)
        if embedding is None:
            print("Failed to get embedding for the query.")
            return []
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
            texts = [hit.entity.get("text") for hit in results[0]]
            return texts
        except Exception as e:
            print(f"An error occurred during the search: {e}")
            return []
        
    def __del__(self):
        self.collection.release()
        print(f"Collection '{self.collection_name}' released.")
        connections.disconnect("default")
        print("Disconnected from Milvus.")


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
            response.raise_for_status()

            full_response = ""
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            full_response += data["response"]
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

            if full_response:
                return full_response
            else:
                return "I'm sorry, I couldn't generate a response."
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return "I'm sorry, I couldn't generate a response."

    def generate_title(self, prompt):
        try:
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json={"model": "llama3.1:8b", "prompt": prompt},
                stream=True
            )
            response.raise_for_status()

            full_response = ""
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            full_response += data["response"]
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

            if full_response:
                return full_response
            else:
                return "I'm sorry, I couldn't generate a response."
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return "I'm sorry, I couldn't generate a response."

# -----------------------------
# UI Classes
# -----------------------------

class LoginUI:
    def __init__(self, master, on_success):
        self.master = master
        self.on_success = on_success
        self.master.title("RagStack - Login")

        # Configure the overall window appearance
        self.master.geometry("700x600")
        self.master.resizable(False, False)
        self.master.configure(bg="#BAA0AC")

        # Style configuration
        style = ttk.Style()
        style.configure("TFrame", background="#BAA0AC")
        style.configure("TLabel", background="#BAA0AC", font=("Arial", 16))
        style.configure("TEntry", font=("Arial", 14))
        style.configure("TButton", font=("Arial", 16), padding=10)
        style.map("TButton", background=[("active", "#A77464")], foreground=[("active", "white")])

        # Main frame
        self.frame = ttk.Frame(self.master, padding=30)
        self.frame.pack(expand=True)

        # Username Label and Entry
        ttk.Label(self.frame, text="Username:").grid(row=0, column=0, pady=10, sticky="e")
        self.username_entry = ttk.Entry(self.frame, width=20)
        self.username_entry.grid(row=0, column=1, pady=10)

        # Password Label and Entry
        ttk.Label(self.frame, text="Password:").grid(row=1, column=0, pady=10, sticky="e")
        self.password_entry = ttk.Entry(self.frame, show="*", width=20)
        self.password_entry.grid(row=1, column=1, pady=10)

        # Login Button
        self.login_button = ttk.Button(self.frame, text="Login", command=self.login)
        self.login_button.grid(row=2, column=0, columnspan=2, pady=20, sticky="ew")

        # Register Button
        self.register_button = ttk.Button(self.frame, text="Register", command=self.register)
        self.register_button.grid(row=3, column=0, columnspan=2, pady=10, sticky="ew")

        # Make buttons fill horizontally
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_columnconfigure(1, weight=1)

    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        data = f"authenticate={username}={password}"
        resp = send_data(data)
        # Handle response
        if len(resp) > 1:
            if resp == "success":
                user_uuid = get_user(username)
                self.on_success(user_uuid)
            else:
                messagebox.showerror("Login Failed", resp)

    def register(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        data = f"register={username}={username}={password}"
        resp = send_data(data)
        if "success" in resp:
            parts = resp.split("=")
            if len(parts) > 1:
                user_uuid = parts[1].strip()
            else:
                user_uuid = str(uuid.uuid4())
            insert_user(username, user_uuid)
            self.on_success(user_uuid)
        else:
            messagebox.showerror("Registration Failed", resp)

class ChatUI:
    def __init__(self, master, retriever, generator, user_uuid):
        self.master = master
        self.retriever = retriever
        self.generator = generator
        self.user_uuid = user_uuid

        self.master.title("RagStack")
        self.master.geometry("865x1000")
        self.master.configure(bg="#BAA0AC")

        # Main container
        self.main_frame = tk.Frame(self.master, bg="#BAA0AC")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Left sidebar
        self.sidebar = tk.Frame(self.main_frame, width=200, bg="#BAA0AC")
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)

        self.home_button = tk.Button(self.sidebar, text="Home", command=self.show_home, bg="#A27E8E", fg="white", font=("Arial", 12))
        self.home_button.pack(pady=10, padx=5, fill=tk.X)

        tk.Label(self.sidebar, text="Chats", bg="#BAA0AC", font=("Arial", 14)).pack(pady=5)
        self.chats_listbox = tk.Listbox(self.sidebar, font=("Arial", 12), bg="#A27E8E", selectbackground="#A27E8E", selectforeground="white")
        self.chats_listbox.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        self.chats_listbox.bind('<<ListboxSelect>>', self.load_chat)

        # Right main area
        self.content_frame = tk.Frame(self.main_frame, bg="#BAA0AC")
        self.content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Show home view initially
        self.show_home()
        self.refresh_chats_list()

    def refresh_chats_list(self):
        self.chats_listbox.delete(0, tk.END)
        chats = get_all_chats(self.user_uuid)
        for cid, title in chats:
            display_name = title if title else f"Chat {cid}"
            self.chats_listbox.insert(tk.END, f"{cid}: {display_name}")

    def show_home(self):
        for w in self.content_frame.winfo_children():
            w.destroy()

        # Home screen: start a new chat
        tk.Label(self.content_frame, text="Welcome to RagStack!", font=("Arial", 36), bg="#BAA0AC").pack(pady=350, padx=10)
        tk.Label(self.content_frame, text="Start a New Chat", font=("Arial", 16), bg="#BAA0AC").pack(pady=20, padx=10)
        self.new_chat_entry = tk.Entry(self.content_frame, font=("Arial", 14), width=40)
        self.new_chat_entry.pack(pady=10)
        self.start_chat_button = tk.Button(self.content_frame, text="Send", font=("Arial", 14), bg="#A27E8E", fg="white", command=self.start_new_chat)
        self.start_chat_button.pack(pady=5)

    def start_new_chat(self):
        user_message = self.new_chat_entry.get().strip()
        if not user_message:
            return

        chat_id = insert_chat(self.user_uuid)
        insert_message(chat_id, "USER", 0, user_message)
        self.show_chat(chat_id)
        print(f"New chat started with ID: {chat_id}")   
        update_chat_title(chat_id, "New Chat")
        self.refresh_chats_list()
        # Retrieve context and generate response
        context_texts = self.retriever.retrieve(user_message)
        if not context_texts:
            bot_response = "I'm sorry, I couldn't find any relevant information."
        else:
            context = "\n".join(context_texts)
            prompt = f"""You are an assistant that provides answers based on the following context.


Context:
{context}

Question: {user_message}
Answer:"""
            bot_response = self.generator.generate_response(prompt)

        insert_message(chat_id, "BOT", 1, bot_response)
        self.refresh_chats_list()
        self.show_chat(chat_id)

        prompt = f"""You are an assistant that will summarize the conversation and give a title to the chat, your awnser should be a title for the chat and no more than 5 words. 

Rules:

DO NOT USE PUNCTUATION MARKS.
DO NOT USE the words Genshin Impact, Chat, or Conversation.

Question: 

{user_message}

response:

{bot_response}

Title:"""
        bot_response = self.generator.generate_title(prompt)

        title = " ".join(bot_response.split()[:5])
        update_chat_title(chat_id, title)
        self.refresh_chats_list()

    def load_chat(self, event):
        selection = self.chats_listbox.curselection()
        if not selection:
            return
        val = self.chats_listbox.get(selection[0])
        # format: "id: title"
        chat_id_str = val.split(":")[0]
        chat_id = int(chat_id_str)
        self.show_chat(chat_id)

    def show_chat(self, chat_id):
        for w in self.content_frame.winfo_children():
            w.destroy()

        def scroll_to_bottom(event):
            self.chat_display.yview_moveto(1.0)

        self.chat_id = chat_id

        self.chat_display_frame = tk.Frame(self.content_frame, bg="#BAA0AC")
        self.chat_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.chat_display = tk.Canvas(self.chat_display_frame, bg="#BAA0AC", highlightthickness=0)
        self.chat_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.chat_scrollbar = tk.Scrollbar(self.chat_display_frame, command=self.chat_display.yview)
        self.chat_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.chat_display.configure(yscrollcommand=self.chat_scrollbar.set)

        self.messages_frame = tk.Frame(self.chat_display, bg="#BAA0AC")
        self.chat_display.create_window((0, 0), window=self.messages_frame, anchor="nw")

        self.messages_frame.bind("<Configure>", lambda e: self.chat_display.configure(scrollregion=self.chat_display.bbox("all")))

        messages = get_chat_messages(chat_id)
        for sender, pos, message in messages:
            self.add_message(sender, message)

        self.chat_display.update_idletasks()  
        self.chat_display.yview_moveto(1.0)


        self.chat_display.bind("<Configure>", scroll_to_bottom)

        self.user_input = tk.Entry(self.content_frame, font=("Arial", 14), width=60)
        self.user_input.pack(pady=10, side=tk.LEFT, padx=10)
        self.user_input.bind("<Return>", self.send_message)

        self.send_button = tk.Button(self.content_frame, text="Send", font=("Arial", 14), bg="#A27E8E", fg="white", command=self.send_message)
        self.send_button.pack(pady=10, side=tk.LEFT)


    def add_message(self, sender, message):
        frame = tk.Frame(self.messages_frame, bg="#A27E8E" if sender == "USER" else "#A77464", pady=5, padx=10)
        frame.pack(anchor="e" if sender == "USER" else "w", fill=tk.NONE, pady=5)

        label = tk.Label(frame, text=message, font=("Arial", 12), bg="#A27E8E" if sender == "USER" else "#A77464", fg="white", wraplength=600, justify="left" if sender == "BOT" else "right")
        label.pack()

    def send_message(self, event=None):
        user_message = self.user_input.get().strip()
        if not user_message:
            return

        self.user_input.delete(0, tk.END)
        self.add_message("USER", user_message)

        pos = get_chat_message_count(self.chat_id)
        insert_message(self.chat_id, "USER", pos, user_message)

        # Retrieve context and generate response
        context_texts = self.retriever.retrieve(user_message)
        if not context_texts:
            bot_response = "I'm sorry, I couldn't find any relevant information."
        else:
            context = "\n".join(context_texts)
            prompt = f"""You are an assistant that provides answers based on the following context.

Context:
{context}

Question: {user_message}
Answer:"""
            bot_response = self.generator.generate_response(prompt)

        pos = get_chat_message_count(self.chat_id)
        insert_message(self.chat_id, "BOT", pos, bot_response)
        self.add_message("BOT", bot_response)

    def __del__(self):
        print("Chat UI destroyed.")

def main():
    # Create local db tables if not exist
    create_user_table()
    create_chats_tables()


    root = tk.Tk()

    def on_login_success(u_uuid):

        # Destroy login frame and show ChatUI
        for w in root.winfo_children():
            w.destroy()
        retriever = Retriever(
            host="127.0.0.1",
            port="19530",
            collection_name="embedded_texts",
            embedding_dim=4096,
        )
        generator = Generator("http://127.0.0.1:11434/api/generate")
        print(f"User UUID: {u_uuid}")
        ChatUI(root, retriever, generator, u_uuid)

    LoginUI(root, on_login_success)
    root.mainloop()

if __name__ == "__main__":
    main()


# TODO
"""
Fix the bug where when you start a new chat the user message is displayed on the right side of the chat. and is only displayed on the left side after the user sends a message 2nd message.

Fix the bug where when show_chat (the update func) is called the chat is not auto scrolled to the bottom. (auto scroll only happens when we load a chat from the sidebar)

Fix the bug where the the Send button is not displayed on the right side of the input field.(in the chat view)

Add a feature to allow string text from the ai API, the api already supports this feature, and we are using it in the Generator class, but the UI does not support it, so we only retrun the whole response as a single string.

"""