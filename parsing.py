from bs4 import BeautifulSoup
from newspaper import Article
import lxml
from readabilipy import simple_json_from_html_string
from goose3 import Goose

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


def get_title_and_text5(html_content) -> (str, str):
    html_str = html_content.decode('utf-8')
    article = simple_json_from_html_string(html_str, use_readability=True)
    title = article.get('title', 'N/A')
    plain_text = article.get('plain_text', [])
    if plain_text:
        text_content = "\n".join([paragraph.get('text', '') for paragraph in plain_text])
    else:
        text_content = "N/A"
    return title, text_content


def get_title_and_text6(html_: bytes) -> (str, str):
    # Ensure the HTML content is decoded
    html_str = html_.decode('utf-8', 'ignore')

    # Use the Goose extractor
    g = Goose()
    article = g.extract(raw_html=html_str)

    # Retrieve title and cleaned text from the extracted article
    title = article.title if article.title else "N/A"
    cleaned_text = article.cleaned_text if article.cleaned_text else "N/A"

    return title, cleaned_text



def save_extracted_txt(filename, title, content):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(f"{title}\n")
        file.write(f"{content}")
    print("{file_name} save completed!")
