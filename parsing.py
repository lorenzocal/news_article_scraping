from bs4 import BeautifulSoup
from newspaper import Article
import lxml
from readabilipy import simple_json_from_html_string
from goose3 import Goose

# get the title and text of the HTML


# get texts with beautifulsoup : extract whole texts
def get_title_and_text1(html: bytes) -> (str, list[str]):
    soup = BeautifulSoup(html, 'html.parser')

    title = soup.find('title').get_text()

    # Get all the text from the HTML
    strings = soup.get_text()

    return title, strings


# get texts with beautifulsoup : extract only 'p tag' text
def get_title_and_text2(html_: bytes) -> (str, str):
    soup = BeautifulSoup(html_, 'html.parser')

    title = soup.find('title').get_text()

    paragraphs = soup.find_all('p')  # Get only p tags strings from within the HTML
    strings = '\n'.join([para.get_text() for para in paragraphs])

    return title, strings


# get texts with newspaper3k
def get_title_and_text3(html_: bytes) -> (str, str):
    article = Article(url='', language='en')
    html_str = html_.decode('utf-8')    # html : byte to str

    article.set_html(html_str)
    article.parse()

    # newspaper3k automatically extract
    title = article.title
    strings = article.text

    return title, strings


# With XPath
def get_title_and_text4(html_: bytes) -> (str, str):
    html_str = html_.decode('utf-8')  # html : byte to str
    tree = lxml.html.fromstring(html_str)

    # title = tree.xpath('//title/text()')[0] # extract title with 'title' tag
    title = tree.xpath('//h1/text()')[0]  # extract title with 'h1' tag

    paragraphs = tree.xpath('//p/text()')  # extract body with 'p' tag
    strings = '\n'.join([para for para in paragraphs])

    return title, strings


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
