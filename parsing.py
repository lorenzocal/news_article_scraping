from bs4 import BeautifulSoup
from newspaper import Article #should install newspaper3k
from lxml import html

## get the title and text of the HTML

# get texts with beautifulsoup : extract whole texts
def get_title_and_text(html: bytes) -> (str, list[str]):
    soup = BeautifulSoup(html, 'html.parser')

    title = soup.find('title').get_text()
    strings = soup.body.strings # Get all strings from within the HTML
    
    return title, strings

# get texts with beautifulsoup : extract only 'p tag' text
def get_title_and_text2(html_: bytes) -> (str, list[str]):
    soup = BeautifulSoup(html_, 'html.parser')

    title = soup.find('title').get_text()

    paragraphs = soup.find_all('p') # Get only p tags strings from within the HTML
    strings = '\n'.join([para.get_text() for para in paragraphs])
    
    return title, strings

# get texts with newspaper3k
def get_title_and_text3(html_: bytes) -> (str, list[str]):
    article = Article(url='', language='en')
    html_str = html_.decode('utf-8')    # html : byte to str

    article.set_html(html_str)
    article.parse()

    # newspaper3k automatically extract
    title = article.title
    strings = article.text

    return title, strings

def get_title_and_text4(html_: bytes) -> (str, list[str]):
    html_str = html_.decode('utf-8')    # html : byte to str
    tree = html.fromstring(html_str)

    #title = tree.xpath('//title/text()')[0] # extract title with 'title' tag
    title = tree.xpath('//h1/text()')[0] # extract title with 'h1' tag
    
    paragraphs = tree.xpath('//p/text()') # extract body with 'p' tag
    strings = '\n'.join([para for para in paragraphs])

    return title, strings
