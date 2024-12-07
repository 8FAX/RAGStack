from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility
from tqdm import tqdm
import requests
import os

def get_field_params(schema, field_name):
    for field in schema.fields:
        if field.name == field_name:
            return field.params
    return None

def truncate_text_to_max_bytes(text, max_bytes):
    encoded_text = text.encode('utf-8')
    if len(encoded_text) <= max_bytes:
        return text
    else:
        truncated_encoded = encoded_text[:max_bytes]
        while True:
            try:
                decoded_text = truncated_encoded.decode('utf-8')
                return decoded_text
            except UnicodeDecodeError:
                truncated_encoded = truncated_encoded[:-1]
                if not truncated_encoded:
                    return ''

class MilvusHandler:
    def __init__(self, host, port, collection_name, embedding_dim):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.collection = None

        self.connect_to_milvus()
        self.initialize_collection()

    def connect_to_milvus(self):
        connections.connect("default", host=self.host, port=self.port)
        print("Connected to Milvus.")

    def initialize_collection(self):
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print(f"Existing collection '{self.collection_name}' has been dropped.")

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
        ]
        schema = CollectionSchema(fields, "Collection for text embeddings")
        self.collection = Collection(name=self.collection_name, schema=schema)
        print(f"Collection {self.collection_name} created with embedding dimension {self.embedding_dim}.")

    def insert_data(self, data):
        try:
            field_params = get_field_params(self.collection.schema, "text")
            max_length = field_params.get("max_length", 1024)

            embeddings = [list(map(float, emb)) for emb in data["embedding"]]
            texts = data["text"]

            insert_data = []
            for emb, txt in zip(embeddings, texts):

                txt = truncate_text_to_max_bytes(txt, max_length)
                insert_data.append({
                    "embedding": emb,
                    "text": txt
                })

            self.collection.insert(insert_data)
            print(f"Successfully inserted {len(embeddings)} records into Milvus.")
        except Exception as e:
            print(f"Error during insertion: {e}")

    def create_index(self):
        self.collection.create_index(
            field_name="embedding",
            index_params={"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 100}},
        )
        print("Index created successfully.")

class TextEmbeddingProcessor:
    def __init__(self, api_url):
        self.api_url = api_url

    def get_embedding(self, text):
        try:
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json={"model": "snowflake-arctic-embed2:latest", "input": text},
            )
            response.raise_for_status()
            data = response.json()
            if "embedding" in data and isinstance(data["embedding"], list):
                return data["embedding"]
            elif "embeddings" in data and isinstance(data["embeddings"], list) and len(data["embeddings"]) > 0:
                return data["embeddings"][0]
            else:
                print("Unexpected API response format.")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None

    @staticmethod
    def split_text_into_chunks(text, max_length=1024, overlap=150):
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = start + max_length
            chunk = text[start:end]

            chunk = truncate_text_to_max_bytes(chunk, max_length)
            chunks.append(chunk)
            start += max_length - overlap
        return chunks

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_files(self):
        files = []
        for file_name in os.listdir(self.data_path):
            file_path = os.path.join(self.data_path, file_name)
            if os.path.isfile(file_path):
                files.append(file_path)
        return files

    @staticmethod
    def read_file(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read().strip()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

class EmbeddingPipeline:
    def __init__(self, milvus_config, embedding_api_url, data_path):
        self.milvus_handler = MilvusHandler(
            milvus_config["host"],
            milvus_config["port"],
            milvus_config["collection_name"],
            milvus_config["embedding_dim"],
        )
        self.embedding_processor = TextEmbeddingProcessor(embedding_api_url)
        self.data_loader = DataLoader(data_path)

    def process_and_insert_data(self):
        files = self.data_loader.load_files()

        field_params = get_field_params(self.milvus_handler.collection.schema, "text")
        max_length = field_params.get("max_length", 1024)

        total_progress = tqdm(files, desc="Total Progress", unit="file")
        for file_path in total_progress:
            text = self.data_loader.read_file(file_path)
            if not text:
                continue

            data_to_insert = {"embedding": [], "text": []}
            chunks = self.embedding_processor.split_text_into_chunks(text, max_length=max_length)
            with tqdm(total=len(chunks), desc=f"Processing {os.path.basename(file_path)}", unit="chunk") as file_progress:
                for chunk in chunks:
                    if not chunk.strip():
                        continue

                    chunk = truncate_text_to_max_bytes(chunk, max_length)

                    embedding = self.embedding_processor.get_embedding(chunk)
                    if embedding and isinstance(embedding, list) and len(embedding) == self.milvus_handler.embedding_dim:
                        data_to_insert["embedding"].append(embedding)
                        data_to_insert["text"].append(chunk)
                    else:
                        tqdm.write(f"Invalid embedding for chunk in file: {file_path}. Skipping.")
                    file_progress.update(1)

            if data_to_insert["embedding"]:
                self.milvus_handler.insert_data(data_to_insert)
            else:
                tqdm.write(f"No valid data to insert for file: {file_path}")

        self.milvus_handler.create_index()


def main():
    
    MILVUS_CONFIG = {
        "host": "127.0.0.1",
        "port": "19530",
        "collection_name": "embedded_texts",
        "embedding_dim": 1024, 
    }
    EMBEDDING_API_URL = "http://127.0.0.1:11434/api/embed"
    DATA_PATH = "./data"

    pipeline = EmbeddingPipeline(MILVUS_CONFIG, EMBEDDING_API_URL, DATA_PATH)
    pipeline.process_and_insert_data()

if __name__ == "__main__":
    main()
