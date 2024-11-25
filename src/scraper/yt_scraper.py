import sys
import time
import threading
import random
import os
import json
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_search import YoutubeSearch
from googleapiclient.discovery import build
import requests
import dotenv


dotenv.load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

OUTPUT_DIR = "data/text_data"
SUMMARY_DIR = "data/summaries"
CACHE_FILE = "data/processed_videos.json"
QUEUE_FILE = "data/queue.json"
CONTEXT_WINDOW = 5000

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

# Stats tracking
stats = {
    "queue_size": 0,
    "avg_queue_entry_time": 0,
    "total_videos": 0,
    "total_summaries": 0,
    "avg_summary_time": 0,
    "total_runtime": 0,
    "iteration_num": 0,
}
errors = []

start_time = time.time()


def print_console_stats():
    """
    The `print_console_stats` function continuously updates the console with live statistics related to
    video processing and summarization.
    
    Author - Liam Scott
    Last update - 11/25/2024
    
    
    """
    while True:
        elapsed_time = time.time() - start_time
        stats["total_runtime"] = elapsed_time

        console_output = "\n".join(errors[-5:])  # Show the last 5 errors
        console_output += "\n" + "-" * 50
        console_output += f"""
Queue Size: {stats['queue_size']}
Average Time per Queue Entry: {stats['avg_queue_entry_time']:.2f}s
Total Videos Processed: {stats['total_videos']}
Total Summaries Created: {stats['total_summaries']}
Average Time per Summary: {stats['avg_summary_time']:.2f}s
Total Runtime: {stats['total_runtime']:.2f}s
Iteration Number: {stats['iteration_num']}
"""
        sys.stdout.write("\033c")  # Clear console
        sys.stdout.write(console_output)
        sys.stdout.flush()
        time.sleep(1)  # Update every second


def load_cache():
    """
    The `load_cache` function reads and returns a set of data from a cache file if it exists, otherwise
    it returns an empty set.
    
    Author - Liam Scott
    Last update - 11/25/2024
    
    
    @ returns A set containing the data loaded from the CACHE_FILE if it exists, otherwise an empty set.
    
    """
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as file:
            return set(json.load(file))
    return set()


def save_cache(cache):
    """
    The function `save_cache` saves the contents of a cache to a file in JSON format.
    
    Author - Liam Scott
    Last update - 11/25/2024
    
    @ param cache ()  - The `cache` parameter is a data structure that stores temporary data in memory,
    typically used to store frequently accessed or computed data to improve performance. In this
    context, the `save_cache` function is responsible for saving the contents of the cache to a file for
    later retrieval or persistence.
    
    """
    with open(CACHE_FILE, 'w') as file:
        json.dump(list(cache), file)


def load_queue():
    """
    The `load_queue` function reads and returns the contents of a JSON file if it exists, otherwise it
    returns an empty list.
    
    Author - Liam Scott
    Last update - 11/25/2024
    
    
    @ returns If the file specified by `QUEUE_FILE` exists, the function will return the contents of the
    file loaded as a JSON object. If the file does not exist, an empty list `[]` will be returned.
    
    """
    if os.path.exists(QUEUE_FILE):
        with open(QUEUE_FILE, 'r') as file:
            return json.load(file)
    return []


def save_queue(queue):
    """
    The function `save_queue` saves a queue to a file using JSON format.
    
    Author - Liam Scott
    Last update - 11/25/2024
    
    @ param queue ()  - The `queue` parameter is a data structure that stores a collection of elements
    in a specific order. In this context, it seems like the `queue` is being saved to a file using the
    `json.dump()` function.
    
    """
    with open(QUEUE_FILE, 'w') as file:
        json.dump(queue, file)


def log_error(message):
    """
    The function `log_error` logs an error message to a list called `errors` and ensures that the list
    does not exceed 100 messages by removing the oldest message if necessary.
    
    Author - Liam Scott
    Last update - 11/25/2024
    
    @ param message ()  - The `message` parameter in the `log_error` function is a string that
    represents the error message that you want to log to the errors list.
    
    """
    """Log an error to the errors list."""
    errors.append(message)
    if len(errors) > 100:
        errors.pop(0)


def download_transcript(video_id):
    """
    The function `download_transcript` attempts to fetch the English transcript of a YouTube video based
    on its video ID, logging errors if any occur.
    
    Author - Liam Scott
    Last update - 11/25/2024
    
    @ param video_id ()  - The `download_transcript` function you provided seems to be a Python function
    that attempts to download the English transcript for a given video ID using the
    `YouTubeTranscriptApi`. If successful, it returns the transcript text; otherwise, it logs an error
    message and returns `None`.
    
    @ returns The function `download_transcript(video_id)` will return the English transcript of the
    video with the specified `video_id` if it is available. If there is no English transcript available
    for the video, it will log an error message and return `None`. If an exception occurs during the
    process of downloading the transcript, it will log an error message with the reason for the
    exception and return `None`.
    
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        for transcript in transcript_list:
            if transcript.language_code == 'en':
                return transcript.fetch()
        log_error(f"No English transcript available for video: {video_id}")
        return None
    except Exception as e:
        log_error(f"Error downloading transcript for video: {video_id}. Reason: {str(e)}")
        return None


def get_video_details(video_id):
    """
    The function `get_video_details` retrieves information about a video from YouTube using its video
    ID.
    
    Author - Liam Scott
    Last update - 11/25/2024
    
    @ param video_id ()  - The `get_video_details` function takes a `video_id` as a parameter. This
    function makes a request to the YouTube API to retrieve details such as title, description, tags,
    views, and likes for a specific video identified by the `video_id`. If successful, it returns a
    dictionary containing
    
    @ returns The function `get_video_details(video_id)` returns a dictionary containing the following
    details of a video with the given `video_id`:
    - "title": The title of the video. If no title is available, it defaults to 'No title available'.
    - "description": The description of the video. If no description is available, it defaults to 'No
    description available'.
    - "tags": A
    
    """
    try:
        request = youtube.videos().list(part="snippet,statistics", id=video_id)
        response = request.execute()

        if not response['items']:
            log_error(f"No details found for video: {video_id}")
            return None

        video_data = response['items'][0]
        snippet = video_data['snippet']
        statistics = video_data['statistics']

        return {
            "title": snippet.get('title', 'No title available'),
            "description": snippet.get('description', 'No description available'),
            "tags": snippet.get('tags', []),
            "views": statistics.get('viewCount', 0),
            "likes": statistics.get('likeCount', 0)
        }

    except Exception as e:
        log_error(f"Error getting video details for video: {video_id}. Reason: {str(e)}")
        return None


def save_as_text(video_id, video_details, transcript, output_dir):
    """
    The function `save_as_text` saves video details and transcript as text in a file and logs any errors
    that occur.
    
    Author - Liam Scott
    Last update - 11/25/2024
    
    @ param video_id ()  - The `video_id` parameter is a unique identifier for the video that will be
    used as part of the file name when saving the text file.
    @ param video_details ()  - The `video_details` parameter is a dictionary containing details about a
    video. It typically includes keys such as 'title', 'description', 'tags', 'views', and 'likes'. The
    'tags' key holds a list of tags associated with the video.
    @ param transcript ()  - The `transcript` parameter in the `save_as_text` function is expected to be
    a list of dictionaries where each dictionary represents a segment of the video transcript. Each
    dictionary should have a key 'text' that contains the text content of that segment.
    @ param output_dir ()  - The `output_dir` parameter in the `save_as_text` function represents the
    directory where the text file will be saved. This directory should be a valid path on the file
    system where the text file will be created.
    
    @ returns The function `save_as_text` returns the file path of the saved text file if the operation
    is successful. If there is an error during the process, it logs the error and returns `None`.
    
    """
    try:
        content_lines = [
            f"Title: {video_details['title']}",
            f"Description: {video_details['description']}",
            f"Tags: {', '.join(video_details.get('tags', []))}",
            f"Views: {video_details['views']}",
            f"Likes: {video_details['likes']}",
            "Transcript:",
        ]
        content_lines.extend([item['text'] for item in transcript])
        content = "\n".join(content_lines)

        file_path = os.path.join(output_dir, f"{video_id}.txt")
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)

        stats["total_videos"] += 1
        return file_path
    except Exception as e:
        log_error(f"Error saving text file for video: {video_id}. Reason: {str(e)}")
        return None


def search_and_download_videos(query, output_dir, max_videos, cached_video_ids, queue):
    """
    The function searches for and downloads videos based on a query, storing them in an output directory
    while managing a queue and cached video IDs.
    
    Author - Liam Scott
    Last update - 11/25/2024
    
    @ param query ()  - The `query` parameter is the search query used to search for videos on YouTube.
    @ param output_dir ()  - The `output_dir` parameter is the directory where the downloaded videos
    will be saved. It is the location on your file system where the downloaded video files will be
    stored.
    @ param max_videos ()  - The `max_videos` parameter specifies the maximum number of videos to search
    for and potentially download based on the given query.
    @ param cached_video_ids ()  - The `cached_video_ids` parameter is a set that stores the video IDs
    of videos that have already been downloaded or processed. This set is used to avoid downloading the
    same video multiple times.
    @ param queue ()  - The `queue` parameter in the `search_and_download_videos` function seems to be a
    list that holds file paths of downloaded videos. It is used to keep track of the videos that have
    been successfully downloaded and saved as text files. The function checks the length of the queue
    and waits if it reaches
    
    """
    try:
        results = YoutubeSearch(query, max_results=max_videos).to_dict()
        random.shuffle(results)

        for video in results:
            while len(queue) >= 100:
                time.sleep(60)  # Wait if queue is full

            video_id = video['id']
            if video_id in cached_video_ids:
                continue

            video_details = get_video_details(video_id)
            if not video_details:
                continue

            transcript = download_transcript(video_id)
            if transcript:
                file_path = save_as_text(video_id, video_details, transcript, output_dir)
                if file_path:
                    queue.append(file_path)
                    cached_video_ids.add(video_id)
                    save_queue(queue)
                    save_cache(cached_video_ids)
                    stats["queue_size"] = len(queue)

    except Exception as e:
        log_error(f"Error searching and downloading videos. Reason: {str(e)}")


def process_queue(queue):
    """
    The function `process_queue` processes files in a queue, splitting them into chunks and summarizing
    each chunk before saving the summaries.
    
    Author - Liam Scott
    Last update - 11/25/2024
    
    @ param queue ()  - The `queue` parameter in the `process_queue` function is a list that contains
    file paths. The function processes each file in the queue by reading its content, splitting it into
    chunks, summarizing each chunk, and then saving the summarized content to a separate file in a
    specified directory. If an
    
    """
    while True:
        if not queue:
            time.sleep(0.1)
            continue

        file_path = queue.pop(0)
        save_queue(queue)

        start_time = time.time()
        try:
            print(f"Processing file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            chunks = split_into_chunks(content, CONTEXT_WINDOW)
            base_filename = os.path.splitext(os.path.basename(file_path))[0]

            for i, chunk in enumerate(chunks):
                summary_start = time.time()
                responce = summarize(chunk)
                summary_end = time.time()

                summary_file = os.path.join(SUMMARY_DIR, f"{base_filename}_{i}_{len(chunks)}.txt")
                with open(summary_file, 'w', encoding='utf-8') as file:
                    file.write(responce)

                stats["total_summaries"] += 1
                total_summary_time = stats["avg_summary_time"] * (stats["total_summaries"] - 1)
                stats["avg_summary_time"] = (total_summary_time + (summary_end - summary_start)) / stats["total_summaries"]

        except Exception as e:
            log_error(f"Error processing file {file_path}. Reason: {str(e)}")

        end_time = time.time()
        total_entry_time = stats["avg_queue_entry_time"] * stats["total_videos"]
        stats["avg_queue_entry_time"] = (total_entry_time + (end_time - start_time)) / stats["total_videos"]
        stats["queue_size"] = len(queue)


def split_into_chunks(text, chunk_size, overlap=500):
    """
    The function `split_into_chunks` divides a given text into chunks of a specified size with a
    specified overlap.
    
    Author - Liam Scott
    Last update - 11/25/2024
    
    @ param text ()  - The `text` parameter is the input text that you want to split into chunks.
    @ param chunk_size ()  - The `chunk_size` parameter specifies the size of each chunk into which the
    text will be split.
    @ param overlap () 500 - The `overlap` parameter in the `split_into_chunks` function determines how
    much overlap there will be between consecutive chunks. In this function, when creating chunks of
    text from the input `text`, the chunks will overlap by the specified number of characters determined
    by the `overlap` parameter. This allows for
    
    @ returns The function `split_into_chunks` returns a list of text chunks, where each chunk has a
    specified size (`chunk_size`) and an optional overlap value (`overlap`).
    
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def summarize(text):
    """
    This function uses an API to generate a summary of a given text related to Genshin Impact videos.
    
    Author - Liam Scott
    Last update - 11/25/2024
    
    @ param text ()  - The `summarize` function takes a text input, sends a request to a local API
    endpoint for text summarization, and returns the summarized response. The `prompt` variable sets up
    the context for the summarization task, including the input text to be summarized. The `Generator`
    class is
    
    @ returns The function `summarize(text)` returns a summary of the input `text` using a text
    generation API for summarizing videos about Genshin Impact.
    
    """
    api_url = "http://127.0.0.1:11434/api/generate" 
    generator = Generator(api_url)

    prompt = f"""You are my assistant, We work to summarize videos about Genshin Impact. Be as concise as possible.

    Video text: {text}

    Summary:"""

    response = generator.generate_response(prompt)
    return response

class Generator:
    def __init__(self, api_url):
        """
        The function is a Python constructor that initializes an object with an API URL.
        
        Author - Liam Scott
        Last update - 11/25/2024
        
        @ param api_url ()  - The `__init__` method is a special method in Python classes used for
        initializing new objects. In this case, the `__init__` method takes `api_url` as a parameter and
        assigns it to the `api_url` attribute of the class instance.
        
        """
        self.api_url = api_url

    def generate_response(self, prompt):
        """
        The function generates a response using a specified prompt by making a POST request to an API
        endpoint and handling the response data.
        
        Author - Liam Scott
        Last update - 11/25/2024
        
        @ param prompt ()  - The code you provided is a method that sends a POST request to an API
        endpoint with a prompt and retrieves a response. The response is then processed to extract the
        generated text.
        
        @ returns The `generate_response` method returns the generated response from the API if
        successful, or "Failed to generate a response." if no response was generated or an error
        occurred during the request.
        
        """
        try:
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json={"model": "llama3.1:8b", "prompt": prompt},
                stream=True
            )
            log_error(f"Response Status Code: {response.status_code}")
            log_error(f"Response Headers: {response.headers}")

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
                    except json.JSONDecodeError as e:
                        log_error(f"JSON decoding failed: {e}")
                        continue

            return full_response if full_response else "Failed to generate a response."
        except requests.exceptions.RequestException as e:
            log_error(f"Request failed: {e}")
            return ""


def main():
    """
    The main function reads topics from a file, processes them in iterations, and searches and downloads
    videos based on the topics.
    
    Author - Liam Scott
    Last update - 11/25/2024
    
    
    """
    topic_file = "input.txt"
    num_iterations = 1000
    cached_video_ids = load_cache()
    queue = load_queue()

    threading.Thread(target=process_queue, args=(queue,), daemon=True).start()
    threading.Thread(target=print_console_stats, daemon=True).start()

    with open(topic_file, 'r') as file:
        topics = [line.strip() for line in file if line.strip()]

    for i in range(num_iterations):
        stats["iteration_num"] = i + 1
        for topic in topics:
            search_and_download_videos(topic, OUTPUT_DIR, max_videos=40, cached_video_ids=cached_video_ids, queue=queue)


if __name__ == "__main__":
    main()

