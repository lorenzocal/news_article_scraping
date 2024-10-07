from bs4 import BeautifulSoup
from newspaper import Article  # should install newspaper3k
from lxml import html
import json
from readabilipy import simple_json_from_html_string


# get the title and text of the HTML

# get texts with beautifulsoup : extract whole texts
def get_title_and_text1(html: bytes) -> (str, list[str]):
    soup = BeautifulSoup(html, 'html.parser')

    title = soup.find('title').get_text()
    strings = soup.body.strings  # Get all strings from within the HTML

    return title, strings


# get texts with beautifulsoup : extract only 'p tag' text
def get_title_and_text2(html_: bytes) -> (str, str):
    soup = BeautifulSoup(html_, 'html.parser')

    title = soup.find('title').get_text()

    paragraphs = soup.find_all('p') # Get only p tags strings from within the HTML
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


def get_title_and_text4(html_: bytes) -> (str, str):
    html_str = html_.decode('utf-8')  # html : byte to str
    tree = html.fromstring(html_str)

    #title = tree.xpath('//title/text()')[0] # extract title with 'title' tag
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


def get_title_and_text_faz(html: bytes) -> (str, list[str]):
    """
    Special version of get_title_and_text for FAZ articles.
    The article title and body text are embedded in a script tag with type "application/ld+json".
    """
    soup = BeautifulSoup(html, 'html.parser')
    scripts = soup.find_all('script', type='application/ld+json')

    for script in scripts:
        # There are multiple script tags with type 'application/ld+json'.
        # The one containing the article title and body text also contains the keys 'headline' and 'articleBody'.
        if 'headline' in script.get_text() and 'articleBody' in script.get_text():
            data = json.loads(script.get_text())
            title = data['headline']
            strings = data['articleBody']
            break
    else:
        title = 'N/A'
        strings = 'N/A'
    return title, strings

def clean_text(text: str) -> str:
    """
    Remove advertisements and other unwanted text from the article text
    """
    # If the only thing there is in a line is "Advertisement" or "Supported by", remove the line
    lines = text.split('\n')
    lines = [line for line in lines if 'Advertisement' not in line and 'Supported by' not in line]

    # Remove sentences like "More on ..." or "More about [name]"
    text = ' '.join(lines)

    return text

def get_title_and_text(url: bytes, html: bytes) -> (str, list[str]):
    """
    Wrapper function to get the title and text of the HTML
    Choose the appropriate function based on the structure of the HTML
    """
    rules = {
        'faz.net': get_title_and_text_faz,
        'www.nytimes.com': get_title_and_text2,
    }
    domain = url.split('/')[2]
    if domain in rules:
        title, text = rules[domain](html)
        return title, clean_text(text)
    else:
        title, text = get_title_and_text2(html)
        return title, clean_text(text)
