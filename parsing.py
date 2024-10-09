import spacy
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

    # Get all the text from the HTML
    strings = soup.get_text()

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


# With XPath
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

def save_extracted_txt(filename, title, content):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(f"{title}\n")
        file.write(f"{content}")
    print("{file_name} save completed!")

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
        # Default to get_title_and_text2
        title, text = get_title_and_text2(html)
        return title, clean_text(text)
    
        # This is an idea but it does not work
        # Try all the functions, and take the one that returns the most text
"""        functions = [get_title_and_text1, get_title_and_text2, get_title_and_text3, get_title_and_text4, get_title_and_text5, get_title_and_text_faz]
        best_text = ''
        best_title = ''
        for function in functions:
            print("Trying function", function)
            title, text = function(html)
            if len(text) > len(best_text):
                best_text = text
                best_title = title
        return best_title, clean_text(best_text)"""

# SEMANTICS

import html

# Convert list of strings to article
def los_to_article(list_):
    return '\n'.join(list_)

def get_title_and_text_semantic(html_content : bytes) -> (str, str):
    """
    Get the title and text of the HTML using semantic similarity
    """

    # Convert the string to bytes
    html_content = html_content.decode('utf-8')
    html_content = html.unescape(html_content)
    soup = BeautifulSoup(html_content, 'html.parser')

    # Get title 
    title = soup.find('title').get_text()

    nlp = spacy.load("en_core_web_md")

    article_so_far = [title]
    nlp_article_so_far = nlp(los_to_article(article_so_far))
    
    # Get all the text from the HTML split into paragraphs
    paragraphs = soup.get_text(separator='\n', strip=True).split('\n')

    # Remove the title from the paragraphs
    paragraphs.remove(title)

    # Remove paragraphs that are too short
    paragraphs = [paragraph for paragraph in paragraphs if len(paragraph) > 10]

    # Iteratively add paragraphs to the article until the similarity between the article and the new paragraph is less than 0.9
    banned_paragraphs = []

    for paragraph in paragraphs:
        nlp_paragraph = nlp(paragraph)
        if nlp_article_so_far.similarity(nlp_paragraph) > 0.7:
            
            article_so_far += [paragraph]
            nlp_article_so_far = nlp(los_to_article(article_so_far))
        else:
            banned_paragraphs.append(paragraph)

    article_so_far.remove(title)

    # Go through the article and remove paragraphs that are not similar enough with the rest of the article
    banned_paragraphs = []
    for paragraph in article_so_far:
        nlp_paragraph = nlp(paragraph)
        if nlp_article_so_far.similarity(nlp_paragraph) < 0.8:
            banned_paragraphs.append(paragraph)
    
    for paragraph in banned_paragraphs:
        article_so_far.remove(paragraph)

    return title, los_to_article(article_so_far)


# Other idea
# Iteratively add paragraphs to the article until the similarity between the article and the article with the new paragraph is less than 0.9
