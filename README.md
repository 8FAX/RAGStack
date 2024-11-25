# YouTube Scraper, Web Scraper, RAG Chatbot, and Milvus Data Loader

## Overview

This repository contains four scripts designed for efficient data processing, retrieval, and summarization:

1. **yt_scraper.py**: Processes YouTube videos by searching, downloading transcripts, summarizing content, and storing metadata.
2. **web_scraper.py**: Extracts content and links from a specified domain, generates summaries, and tracks progress.
3. **app.py**: Implements a Retrieval-Augmented Generation (RAG) chatbot using Milvus for context retrieval and a custom API for generating responses.
4. **load_db.py**: Processes text files, generates embeddings, and inserts them into a Milvus vector database for efficient retrieval.

---

## Table of Contents

1. [YouTube Scraper (`yt_scraper.py`)](#youtube-scraper)
    - [Setup and Configuration](#yt-setup-and-configuration)
    - [Directory Structure](#yt-directory-structure)
    - [Key Functions and Classes](#yt-key-functions-and-classes)
    - [Workflow and Execution](#yt-workflow-and-execution)
2. [Web Scraper (`web_scraper.py`)](#web-scraper)
    - [Setup and Configuration](#web-setup-and-configuration)
    - [Directory Structure](#web-directory-structure)
    - [Key Functions and Classes](#web-key-functions-and-classes)
    - [Workflow and Execution](#web-workflow-and-execution)
3. [RAG Chatbot (`app.py`)](#rag-chatbot)
    - [Components](#components)
    - [Setup and Configuration](#rag-setup-and-configuration)
    - [Workflow](#rag-workflow)
4. [Milvus Data Loader (`load_db.py`)](#milvus-data-loader)
    - [Components](#milvus-components)
    - [Setup and Configuration](#milvus-setup-and-configuration)
    - [Workflow](#milvus-workflow)
5. [How to Run](#how-to-run)
6. [Future Enhancements](#future-enhancements)

---

## YouTube Scraper

### Setup and Configuration

#### Required Libraries:
- `sys`, `time`, `threading`, `random`, `os`, `json`, `dotenv`, `requests`
- `youtube_transcript_api`
- `youtube_search`
- `googleapiclient.discovery`

#### Environment Variables:
- `YOUTUBE_API_KEY`: Set in `.env` file for YouTube API authentication.

#### Configuration Variables:
- `OUTPUT_DIR`: Directory for storing text data.
- `SUMMARY_DIR`: Directory for saving summaries.
- `CACHE_FILE`: File for processed video IDs.
- `QUEUE_FILE`: File for managing queued tasks.
- `CONTEXT_WINDOW`: Maximum context size for summarization chunks.

#### Directories Initialization:
```python
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)
```

---

### Directory Structure

```
data/
├── text_data/         # Stores text files with video details and transcripts.
├── summaries/         # Stores generated summaries.
├── processed_videos.json  # Tracks processed video IDs.
├── queue.json         # Tracks the current processing queue.
```

---

### Key Functions and Classes

#### Utility Functions:
- `print_console_stats()`: Displays live statistics.
- `load_cache() / save_cache()`: Manages processed video cache.
- `load_queue() / save_queue()`: Manages the processing queue.
- `log_error()`: Logs errors to a rotating list.

#### Video Handling:
- `download_transcript(video_id)`: Fetches English transcripts.
- `get_video_details(video_id)`: Retrieves video metadata.
- `save_as_text(video_id, video_details, transcript, output_dir)`: Saves metadata and transcripts.
- `search_and_download_videos(query, output_dir, max_videos, cached_video_ids, queue)`: Searches and downloads videos.

#### Queue Management:
- `process_queue(queue)`: Processes queued files, splits text into chunks, summarizes them, and saves summaries.

#### Summarization Helpers:
- `split_into_chunks(text, chunk_size, overlap=500)`: Splits text into chunks.
- `summarize(text)`: Summarizes content using a custom API.

---

### Workflow and Execution

1. Reads topics from `input.txt`.
2. Initializes caches and processing queue.
3. Launches background threads:
   - **Queue Processor**: Processes tasks in the queue.
   - **Statistics Display**: Updates console stats.
4. Iteratively:
   - Searches videos for topics.
   - Downloads metadata and transcripts.
   - Queues files for summarization.

---

## Web Scraper

### Setup and Configuration

#### Required Libraries:
- `threading`, `requests`, `bs4`, `os`, `time`, `json`, `sys`, `re`, `dotenv`

#### Environment Variables:
- `BASE_URL`: Root URL for scraping.
- `UNWANTED_PATH_SEGMENTS`: Path segments to exclude.

#### Configuration Variables:
- `OUTPUT_DIR`: Directory for raw text.
- `SUMMARY_DIR`: Directory for summaries.
- `CACHE_FILE`: File for visited URLs.
- `QUEUE_FILE`: File for task management.
- `CONTEXT_WINDOW`: Character limit for content chunks.

---

### Directory Structure

```
data/website/
├── text_data/         # Stores scraped content.
├── summaries/         # Stores generated summaries.
├── processed_videos.json  # Tracks visited URLs.
├── queue.json         # Tracks the current processing queue.
```

---

### Key Functions and Classes

#### Utility Functions:
- `is_same_domain(link, base_domain)`: Checks if a link belongs to the same domain.
- `normalize_url(url)`: Removes fragments from a URL.
- `filter_links_by_segments(links, base_domain, unwanted_segments)`: Filters unwanted links.
- `load_cache() / save_cache()`: Manages visited URLs.
- `load_queue() / save_queue()`: Manages the processing queue.
- `log_error()`: Logs errors.

#### Scraping Functions:
- `get_all_links_and_text(url, base_domain)`: Fetches content and links.
- `scrape_domain(base_url, output_dir, queue, max_pages=100, unwanted_segments=None, cache=None)`: Recursively scrapes pages.

---

### Workflow and Execution

1. Loads configuration from `.env`.
2. Initializes caches and queue.
3. Launches background threads for processing queue and stats.
4. Calls `scrape_domain()` to:
   - Extract links and content.
   - Save content and queue tasks.

---

## RAG Chatbot

### Components

#### Core Libraries:
- `pymilvus`, `requests`, `tkinter`, `json`

#### Core Functionalities:
1. **Retriever**: Fetches context from Milvus.
2. **Generator**: Generates responses using a custom API.
3. **ChatbotUI**: Interactive chatbot GUI.

---

### Setup and Configuration

#### Milvus:
- Host: `127.0.0.1`
- Port: `19530`
- Collection: `embedded_texts`
- Dimension: `4096`

#### API Endpoints:
- Embedding API: `http://127.0.0.1:11434/api/embed`
- Generation API: `http://127.0.0.1:11434/api/generate`

---

### Workflow

1. **User Input**: User enters a query.
2. **Context Retrieval**:
   - Query is embedded and sent to Milvus.
   - Retrieves relevant texts.
3. **Response Generation**:
   - Constructs a prompt using the context.
   - Sends to Generation API.
4. **Display Response**: Shows in the GUI.

---

## Milvus Data Loader

### Components

#### Core Libraries:
- `pymilvus`, `requests`, `os`, `tqdm`

#### Core Functionalities:
1. **MilvusHandler**: Manages Milvus connections and data insertion.
2. **TextEmbeddingProcessor**: Generates embeddings and splits text.
3. **DataLoader**: Handles file operations.
4. **EmbeddingPipeline**: Orchestrates data loading and insertion.

---

### Setup and Configuration

#### Milvus:
- Host: `127.0.0.1`
- Port: `19530`
- Collection: `embedded_texts`
- Dimension: `4096`

#### API Configuration:
- Embedding API: `http://127.0.0.1:11434/api/embed`

#### Data Path:
- Place text files in `./data`.

---

### Workflow

1. Load text files.
2. Split text into chunks.
3. Generate embeddings using the API.
4. Insert data into Milvus.
5. Create an index for efficient retrieval.

---

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start required services (e.g., Milvus).
3. Configure `.env` files.
4. Run the desired script:
   ```bash
   python yt_scraper.py
   python web_scraper.py
   python app.py
   python load_db.py
   ```

---

## Future Enhancements

- **Error Recovery**: Retry failed API calls.
- **Scalability**: Batch processing for large datasets.
- **Multi-Collection Support**: Handle multiple datasets dynamically.
- **Improved Indexing**: Support advanced index types for Milvus.