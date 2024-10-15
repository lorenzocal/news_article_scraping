import requests
from bs4 import BeautifulSoup
import json
import re

# CHECK: the next two functions are repeated in the fetching file
def get_url_list() -> list:
    """
    Get the list of URLs from the JSON file.

    Returns:
    list: A list of URLs to scrape.
    """
    path = './data/url_list.json'
    with open(path, 'r') as f:
        data = json.load(f)

    return data['urls']

def get_article_simple(url) -> bytes:
    """
    Get the HTML of the article from the URL.
    Args:
        url: url of the page to scrape

    Returns:
        bytes: The HTML of the article.
    """

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Referer': 'https://www.google.com/'
    }
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Failed to load page {url}. Error: {response.status_code}")

    html = response.content

    return html

# Returns True if the input is JSON structured data, False otherwise
def is_json(text):
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False

# Decomposes the JSON returning a list of all its elements
def get_json_elements(data):

    all_elements = []

    # Check if the data is a dictionary
    if isinstance(data, dict):
        for key, value in data.items():
            # Recursively add each value
            all_elements.append(value)
    # Check if the data is a list
    elif isinstance(data, list):
        for value in data:
            # Recursively add each element in the list
            all_elements.append(value)
    else:
        # Base case: if it's not a dict or list, it's a value
        all_elements.append(data)

    return all_elements


# Computes the number of common words between two strings
def common_words_percentage(str1, str2):
    # Tokenize the strings into words and convert them into sets
    words1 = set(str1.split())
    words2 = set(str2.split())
    
    # Find the number of common words
    common_words = words1.intersection(words2)
    
    # Get the length of the shorter string's word set
    shortest_len = min(len(words1), len(words2))
    
    # Calculate percentage of common words relative to the shorter string
    percentage_common = len(common_words) / shortest_len if shortest_len > 0 else 0
    
    return percentage_common

# Removes a string if it shares almost all its words with another string in the list (the shorter string is removed)
def remove_duplicates(strings):
    to_remove = set()

    # Compare each string with every other string in the list
    for i in range(len(strings)):
        for j in range(i + 1, len(strings)):
            str1 = strings[i]
            str2 = strings[j]
            
            # Check if more than 90% of the words in the shorter string are common
            if common_words_percentage(str1, str2) > 0.9:
                # Remove the shorter string
                if len(str1.split()) < len(str2.split()):
                    to_remove.add(str1)
                else:
                    to_remove.add(str2)

    # Filter out the strings to remove
    filtered_strings = [s for s in strings if s not in to_remove]
    return filtered_strings

def extract_text_json(soup):
  
  extracted_text = []

  for element in soup.find_all(True):
    if (is_json(element.text)):

      # Parse the JSON string into a Python dictionary
      data = json.loads(element.text)

      # Cycle over the elements of the JSON
      for element in get_json_elements(data):
        try:
          string = str(element).strip()
          # Remove HTML tags from the textual content
          string = re.sub(r'<[^>]*>', '', string)
          # Check if the line looks like a sentence (starts with a capital letter, has a minimum lenght and ends with a period, exclamation, or question mark)
          if len(string) > 50 and string[0].isupper() and string[-1] in '.!?' and string not in extracted_text:
              extracted_text.append(string)
        except Exception:
          pass 
        
  return " ".join(remove_duplicates(extracted_text))

#### Main ####
html_content = get_article_simple('https://www.zdnet.com/article/where-ai-avatars-are-at-your-service-247/')
soup = BeautifulSoup(html_content, 'lxml')
text = extract_text_json(soup)

print(text)
