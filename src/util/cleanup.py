import os
import time
from pathlib import Path

def check_and_remove_blank_files(folder_path, junk_file_path):
    try:
        with open(junk_file_path, 'a') as junk_file:
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
                    file_id = os.path.splitext(filename)[0]
                    junk_file.write(file_id + '\n')
                    os.remove(file_path)
                    print(f"Removed empty file: {filename}")
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    folder_path = input("Enter the path to the folder to check: ")
    junk_file_path = input("Enter the path to the master junk file: ")
    check_and_remove_blank_files(folder_path, junk_file_path)


def process_files(folder_path):
    """Process files in a folder, removing those with unwanted domains."""
    unwanted_domains = {"i", "pt-br", "ru"}
    file_count = 0
    removed_files = 0

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if not os.path.isfile(file_path):
            continue

        for _ in range(5): 
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    first_line = file.readline().strip()

                if not first_line.startswith("URL:"):
                    print(f"Skipping file {filename}: No valid URL in the first line.")
                    break

                url = first_line.replace("URL:", " URL HERE").strip()
                if not url.startswith(""):
                    print(f"Skipping file {filename}: URL does not match base domain.")
                    break

                relative_path = url.replace(" URL HERE ", "")
                domain_segment = relative_path.split('/')[0]

                if domain_segment in unwanted_domains:
                    os.remove(file_path)
                    removed_files += 1
                    print(f"Removed file {filename}: Unwanted domain '{domain_segment}'.")
                else:
                    file_count += 1

                break 

            except PermissionError:
                print(f"File {filename} is locked. Retrying...")
                time.sleep(1)  
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                break

    print(f"Processing complete. {file_count} files kept, {removed_files} files removed.")



if __name__ == "__main__":

    folder_path = "scraped_data"
    process_files(folder_path)
