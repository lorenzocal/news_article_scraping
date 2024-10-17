# ###### ####### ######  ####
#!pip install newspaper3k   #
#!pip install readabilipy   #
#!pip install goose3        #
# ###### ####### ######  ####

from bs4 import BeautifulSoup
from newspaper import Article
import lxml
from readabilipy import simple_json_from_html_string
from goose3 import Goose
import pandas as pd
import json
import re
# get the title and text of the HTML

# get texts with beautifulsoup : extract whole texts
def get_title_and_text1(html: bytes) -> (str, list[str]):
    # Parse the HTML content
    soup = BeautifulSoup(html, 'html.parser')

    # Extract the title text
    title = soup.find('title').get_text() if soup.find('title') else 'N/A'

    # Get all text and split it into a list of strings by lines
    text = soup.get_text()
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    return title, lines


# get texts with beautifulsoup : extract only 'p tag' text
def get_title_and_text2(html_: bytes) -> (str, list[str]):
    # Parse the HTML content
    soup = BeautifulSoup(html_, 'html.parser')

    # Extract the title text
    title = soup.find('title').get_text() if soup.find('title') else 'N/A'

    # Get all <p> tag strings from within the HTML and return as a list
    paragraphs = soup.find_all('p')
    lines = [para.get_text() for para in paragraphs]

    return title, lines


# get texts with newspaper3k
def get_title_and_text3(html_: bytes) -> (str, list[str]):
    article = Article(url='')
    html_str = html_.decode('utf-8')  # Convert bytes to string

    article.set_html(html_str)
    article.parse()

    # Extract the title
    title = article.title

    # Split text into lines
    lines = article.text.splitlines()

    return title, lines


def get_title_and_text4(html_: bytes) -> (str, list[str]):
    # Convert bytes to string with error handling for encoding issues
    html_str = html_.decode('utf-8', errors='ignore')
    
    # Parse the HTML content
    tree = lxml.html.fromstring(html_str)

    # Extract the title using 'h1' tag, handle cases where 'h1' might be missing
    title = tree.xpath('//h1/text()')
    title = title[0].strip() if title else "N/A"

    # Extract text from 'p' tags, strip whitespace for each paragraph
    lines = [line.strip() for line in tree.xpath('//p/text()') if line.strip()]

    return title, lines


def get_title_and_text5(html_content: bytes) -> (str, list[str]):

    # Convert bytes to string and handle decoding issues
    html_str = html_content.decode('utf-8')

    # Parse the HTML content into a JSON-like structure using readability
    article = simple_json_from_html_string(html_str, use_readability=True)

    # Extract title, with a default of 'N/A' if not found
    title = article.get('title', 'N/A')
    
    # Extract the text content from plain_text, default to an empty list if not found
    plain_text = article.get('plain_text', [])
    if plain_text:
        lines = [paragraph.get('text', '') for paragraph in plain_text]
    else:
        lines = ["N/A"]

    # Remove duplicate lines
    lines = pd.Series(lines).drop_duplicates().tolist()

    return title, lines


def get_title_and_text6(html_: bytes) -> (str, list[str]):
    # Ensure the HTML content is decoded
    html_str = html_.decode('utf-8', 'ignore')

    # Use the Goose extractor
    g = Goose()
    article = g.extract(raw_html=html_str)

    # Retrieve title with a default value if missing
    title = article.title if article.title else "N/A"
    
    # Split cleaned text into paragraphs (assumes paragraphs are separated by new lines)
    if article.cleaned_text:
        lines = article.cleaned_text.split('\n')  # Split by new line
        # Strip whitespace and filter out empty strings
        lines = [line.strip() for line in lines if line.strip()]
    else:
        lines = ["N/A"]

    return title, lines

def get_title_and_text7(soup):

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

  # Convert bytes to string and handle decoding issues
  html_str = html_content.decode('utf-8')

  # Parse the HTML content into a JSON-like structure using readability
  article = simple_json_from_html_string(html_str, use_readability=True)

  # Extract title, with a default of 'N/A' if not found
  title = article.get('title', 'N/A')

  lines = []
  soup = BeautifulSoup(html_content, 'lxml')


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
          if len(string) > 50 and string[0].isupper() and string[-1] in '.!?' and string not in lines:
              lines.append(string)
        except Exception:
          pass

  return title, lines

def save_extracted_txt(filename, title, content):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(f"{title}\n")
        file.write(f"{content}")
    print("{file_name} save completed!")
