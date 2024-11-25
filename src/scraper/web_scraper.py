import threading
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
import os
import time
import json
import sys
import re
import dotenv

# Constants
OUTPUT_DIR = "data/keqingmains/text_data"
SUMMARY_DIR = "data/keqingmains/summaries"
CACHE_FILE = "data/keqingmains/processed_videos.json"
QUEUE_FILE = "data/keqingmains/queue.json"
CONTEXT_WINDOW = 10000

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

# Stats tracking
stats = {
    "queue_size": 0,
    "avg_queue_entry_time": 0,
    "total_pages": 0,
    "total_summaries": 0,
    "avg_summary_time": 0,
    "total_runtime": 0
}
errors = []

start_time = time.time()

def is_same_domain(link, base_domain):
    """
    The function `is_same_domain` checks if a given link belongs to the same domain as a specified base
    domain.
    
    Author - Liam Scott
    Last update - 11/25/2024
    
    @ param link ()  - The `link` parameter is the URL of the link that you want to check if it belongs
    to the same domain as the `base_domain`.
    @ param base_domain ()  - The `base_domain` parameter should be a string representing the base
    domain you want to compare the link to. It should include the protocol (e.g., "https://") and the
    domain name (e.g., "example.com").
    
    @ returns The function `is_same_domain` is returning a boolean value indicating whether the provided
    `link` belongs to the same domain as the `base_domain`.
    
    """
    parsed_base = urlparse(base_domain)
    parsed_link = urlparse(link)
    return parsed_link.netloc == parsed_base.netloc or parsed_link.netloc == ''

def normalize_url(url):
    """
    The `normalize_url` function removes the fragment identifier from a URL.
    
    Author - Liam Scott
    Last update - 11/25/2024
    
    @ param url ()  - The `normalize_url` function takes a URL as input and returns the normalized URL
    after removing any fragment identifiers.
    
    @ returns The function `normalize_url` returns the normalized URL after removing any fragment
    identifier using the `urldefrag` function.
    
    """
    normalized_url, _ = urldefrag(url)  
    return normalized_url

def filter_links_by_segments(links, base_domain, unwanted_segments):
    """
    The function filters a set of links by excluding those that belong to the same domain as the base
    domain and contain any unwanted segments in their paths.
    
    Author - Liam Scott
    Last update - 11/25/2024
    
    @ param links ()  - The `links` parameter is a list of URLs that you want to filter based on certain
    criteria. Each URL in the list represents a link that you want to analyze and potentially include or
    exclude from the final filtered set.
    @ param base_domain ()  - The `base_domain` parameter is a string representing the base domain that
    you want to filter the links by. This is typically the domain that you are interested in, and you
    want to keep only the links that belong to this domain.
    @ param unwanted_segments ()  - The `unwanted_segments` parameter is a list of segments that you do
    not want to appear in the path of the links. The function `filter_links_by_segments` takes a set of
    links, a base domain, and a list of unwanted segments as input. It filters out the links that belong
    
    @ returns The function `filter_links_by_segments` returns a set of links that belong to the same
    domain as the base domain and do not contain any unwanted segments in their path.
    
    """
    filtered_links = set()
    for link in links:
        if not is_same_domain(link, base_domain):
            continue
        parsed_link = urlparse(link)
        path_segments = parsed_link.path.split('/')  
        if not any(segment in unwanted_segments for segment in path_segments):
            filtered_links.add(link)
    return filtered_links

def get_all_links_and_text(url, base_domain):
    """
    The function `get_all_links_and_text` scrapes the text content and all links from a given URL, with
    error handling included.
    
    Author - Liam Scott
    Last update - 11/25/2024
    
    @ param url ()  - The `url` parameter is the URL of the webpage from which you want to extract links
    and text content. It is the webpage that you want to scrape for links and text.
    @ param base_domain ()  - The `base_domain` parameter in the `get_all_links_and_text` function is
    used to specify the base domain against which relative URLs should be resolved. This is important
    when extracting links from a webpage because some links may be specified as relative URLs (e.g.,
    `/about`), and we need
    
    @ returns The function `get_all_links_and_text` returns two values: `text_content` and `links`.
    
    """
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            log_error(f"Failed to fetch {url}. Status code: {response.status_code}")
            return None, None
        
        soup = BeautifulSoup(response.text, 'html.parser')

        text_content = soup.get_text(separator='\n', strip=True)

        links = set()
        for tag in soup.find_all('a', href=True):
            absolute_link = urljoin(url, tag['href'])
            normalized_link = normalize_url(absolute_link)
            links.add(normalized_link)
        
        return text_content, links

    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return None, None

def scrape_domain(base_url, output_dir, queue, max_pages=100, unwanted_segments=None, cache=None):
    """
    This Python function scrapes web pages within a specified domain, saves cleaned content to text
    files, and manages a queue of pages to visit.
    
    Author - Liam Scott
    Last update - 11/25/2024
    
    @ param base_url ()  - The `base_url` parameter in the `scrape_domain` function is the starting URL
    from which the web scraping process will begin. It serves as the root URL for crawling and
    extracting content from web pages within the same domain.
    @ param output_dir ()  - The `output_dir` parameter in the `scrape_domain` function is the directory
    where the scraped content will be saved as text files. It is the location where the cleaned content
    from each URL will be stored in a text file format.
    @ param queue ()  - The `queue` parameter in the `scrape_domain` function seems to be used for
    storing file paths of the scraped content. It is initially passed as an empty list and then files
    are appended to it as the function scrapes and saves content from different URLs. The queue is also
    saved and loaded
    @ param max_pages () 100 - The `max_pages` parameter in the `scrape_domain` function specifies the
    maximum number of pages to scrape before stopping the scraping process. If the number of pages
    scraped reaches the `max_pages` limit, the function will stop scraping and return the results.
    @ param unwanted_segments ()  - The `unwanted_segments` parameter in the `scrape_domain` function is
    a list that contains segments of URLs that should be excluded during the scraping process. These
    segments are used to filter out certain links that match the unwanted criteria. By providing a list
    of unwanted segments, you can ensure that the
    @ param cache ()  - The `cache` parameter in the `scrape_domain` function is used to store the URLs
    that have already been visited and scraped. This helps in avoiding revisiting the same URLs and
    ensures that each URL is processed only once. The cache is a set data structure that stores unique
    URLs.
    
    @ returns The function `scrape_domain` returns the updated `cache` after scraping and processing
    URLs within the specified limits and conditions.
    
    """
    if unwanted_segments is None:
        unwanted_segments = []

    while len(queue) >= 100:
                time.sleep(60)


    if cache is None:
        cache = load_cache()

    to_visit = set([normalize_url(base_url)])
    os.makedirs(output_dir, exist_ok=True)

    while to_visit and len(cache) < max_pages:
        current_url = to_visit.pop()
        if current_url in cache:
            continue

        log_error(f"Scraping: {current_url}")
        text_content, links = get_all_links_and_text(current_url, base_url)

        if text_content:
            pattern = r"Skip.*?Русский"
            cleaned_content = re.sub(pattern, "", text_content, flags=re.DOTALL)

            file_name = os.path.join(output_dir, f"{len(cache)}.txt")
            with open(file_name, 'w', encoding='utf-8') as file:
                file.write(f"URL: {current_url}\n\n{cleaned_content}")
            log_error(f"Saved cleaned content from {current_url} to {file_name}")

            stats["total_pages"] += 1
            queue.append(file_name)
            save_queue(queue)

        if links:
            filtered_links = filter_links_by_segments(links, base_url, unwanted_segments)
            to_visit.update(filtered_links - cache)

        cache.add(current_url)
        save_cache(cache)

        time.sleep(1) 

    log_error("Scraping complete.")
    log_error(f"Visited {len(cache)} pages.")
    return cache

def summarize(text):
    """
    This Python function uses an API to generate a summary of text related to Genshin Impact.
    
    Author - Liam Scott
    Last update - 11/25/2024
    
    @ param text ()  - The `summarize` function takes a text input and uses a Generator class to
    generate a summary of the text. The function first defines an API URL and initializes a Generator
    object with that URL. It then creates a prompt using the input text and sends this prompt to the
    generator to get a summary
    
    @ returns The function `summarize(text)` returns a summarized version of the input text about
    Genshin Impact by utilizing a text generation API.
    
    """
    api_url = "http://127.0.0.1:11434/api/generate" 
    generator = Generator(api_url)

    prompt = f"""You are my assistant, We work to summarize wikis about Genshin Impact. Be as concise as possible. make sure to include the main points and key details in grate detail.

    page text: {text}

    Summary:"""

    response = generator.generate_response(prompt)
    return response

class Generator:
    def __init__(self, api_url):
        """
        The above function is a Python constructor that initializes an object with an API URL.
        
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
        endpoint and processing the received data.
        
        Author - Liam Scott
        Last update - 11/25/2024
        
        @ param prompt ()  - The code you provided seems to be a method for generating a response using
        an API. The `generate_response` method sends a POST request to a specified API URL with a given
        prompt, retrieves the response data, and returns the full response generated by the API.
        
        @ returns The function `generate_response` returns the generated response from the API based on
        the provided prompt. If a response is successfully generated, it returns the full response text.
        If no response is generated or an error occurs during the process, it returns the message
        "Failed to generate a response."
        
        """
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
                    except json.JSONDecodeError as e:
                        log_error(f"JSON decoding failed: {e}")
                        continue

            return full_response if full_response else "Failed to generate a response."
        except requests.exceptions.RequestException as e:
            log_error(f"Request failed: {e}")
            return ""

def process_queue(queue):
    """
    The `process_queue` function processes files in a queue, summarizes their content, and logs any
    errors encountered.
    
    Author - Liam Scott
    Last update - 11/25/2024
    
    @ param queue ()  - The `queue` parameter in the `process_queue` function is a list that contains
    file paths. The function processes each file in the queue by reading its content, splitting it into
    chunks, summarizing each chunk, and then saving the summarized content to separate files. It also
    keeps track of various statistics
    
    """
    while True:
        if not queue:
            time.sleep(0.1)
            continue

        file_path = queue.pop(0)
        save_queue(queue)

        start_time = time.time()
        try:
            log_error(f"Processing file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            chunks = split_into_chunks(content, CONTEXT_WINDOW)
            base_filename = os.path.splitext(os.path.basename(file_path))[0]

            for i, chunk in enumerate(chunks):
                summary_start = time.time()
                try:
                    responce = summarize(chunk)
                    summary_end = time.time()
                except Exception as e:
                    log_error(f"Error summarizing chunk {i} from {file_path}. Reason: {str(e)}")
                    continue

                summary_file = os.path.join(SUMMARY_DIR, f"{base_filename}_{i}_{len(chunks)}.txt")
                with open(summary_file, 'w', encoding='utf-8') as file:
                    file.write(responce)

                stats["total_summaries"] += 1
                total_summary_time = stats["avg_summary_time"] * (stats["total_summaries"] - 1)
                stats["avg_summary_time"] = (total_summary_time + (summary_end - summary_start)) / stats["total_summaries"]
        except Exception as e:
            log_error(f"Error processing file {file_path}. Reason: {str(e)}")
        finally:
            end_time = time.time()
            total_entry_time = stats["avg_queue_entry_time"] * stats["queue_size"]

            if stats["queue_size"] > 0:
                stats["avg_queue_entry_time"] = (total_entry_time + (end_time - start_time)) / stats["queue_size"]
            stats["queue_size"] = len(queue)

def split_into_chunks(text, chunk_size, overlap=1000):
    """
    The function `split_into_chunks` takes a text and splits it into chunks of a specified size with a
    specified overlap.
    
    Author - Liam Scott
    Last update - 11/25/2024
    
    @ param text ()  - The `text` parameter is the input text that you want to split into chunks.
    @ param chunk_size ()  - The `chunk_size` parameter specifies the size of each chunk into which the
    text will be split.
    @ param overlap () 1000 - The `overlap` parameter in the `split_into_chunks` function determines how
    much overlap there will be between consecutive chunks. In this function, the chunks are created by
    moving a window of size `chunk_size` over the input `text`, and the overlap specifies how much of
    the previous chunk will be
    
    @ returns The function `split_into_chunks` returns a list of text chunks, where each chunk has a
    size of `chunk_size` characters with an overlap of `overlap` characters between consecutive chunks.
    
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def print_console_stats():
    """
    The function `print_console_stats` continuously displays real-time statistics related to a
    processing queue on the console.
    
    Author - Liam Scott
    Last update - 11/25/2024
    
    
    """
    while True:
        elapsed_time = time.time() - start_time
        stats["total_runtime"] = elapsed_time

        console_output = "\n".join(errors[-10:]) 
        console_output += "\n" + "-" * 50
        console_output += f"""
Queue Size: {stats['queue_size']}
Average Time per Queue Entry: {stats['avg_queue_entry_time']:.2f}s
Total pages Processed: {stats['total_pages']}
Total Summaries Created: {stats['total_summaries']}
Average Time per Summary: {stats['avg_summary_time']:.2f}s
Total Runtime: {stats['total_runtime']:.2f}s
"""
        sys.stdout.write("\033c")
        sys.stdout.write(console_output)
        sys.stdout.flush()
        time.sleep(1) 

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
    
    @ param cache ()  - The `cache` parameter is a data structure that stores temporary data that can be
    quickly accessed when needed. In this context, it seems like the `cache` is being saved to a file
    using the `save_cache` function.
    
    """
    with open(CACHE_FILE, 'w') as file:
        json.dump(list(cache), file)

def load_queue():
    """
    The `load_queue` function reads and returns the contents of a JSON file if it exists, otherwise it
    returns an empty list.
    
    Author - Liam Scott
    Last update - 11/25/2024
    
    
    @ returns A list is being returned. If the QUEUE_FILE exists, the function will load the contents of
    the file as JSON and return it. Otherwise, an empty list will be returned.
    
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
    in a specific order, typically following the First In First Out (FIFO) principle. It can be
    implemented using various data structures such as lists, arrays, or linked lists. In the context of
    the `save_queue`
    
    """
    with open(QUEUE_FILE, 'w') as file:
        json.dump(queue, file)

def log_error(message):
    """
    The `log_error` function appends error messages to a list and logs them to a file, limiting the list
    to 100 messages.
    
    Author - Liam Scott
    Last update - 11/25/2024
    
    @ param message ()  - The `log_error` function takes a `message` parameter, which is the error
    message that needs to be logged. This message will be appended to the `errors` list and also written
    to a log file named "errors_log.txt" along with a timestamp.
    
    """
    errors.append(message)
    if len(errors) > 100:
        errors.pop(0)
    # Append the error message to a log file
    #with open("errors_log.txt", "a", encoding="utf-8") as error_file:
      #  error_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

if __name__ == "__main__":
    dotenv.load_dotenv()


    base_url = os.getenv("BASE_URL")
    unwanted_path_segments = os.getenv("UNWANTED_PATH_SEGMENTS").split(",") if os.getenv("UNWANTED_PATH_SEGMENTS") else []

    queue = load_queue()
    cache = load_cache()

    threading.Thread(target=process_queue, args=(queue,), daemon=True).start()
    threading.Thread(target=print_console_stats, daemon=True).start()

    scraped_links = scrape_domain(base_url, OUTPUT_DIR, queue, max_pages=10000, unwanted_segments=unwanted_path_segments, cache=cache)

