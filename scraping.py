import requests
from bs4 import BeautifulSoup
import json

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
        # 'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Referer': 'https://www.google.com/'
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Failed to load page {url}. Error: {response.status_code}")
    
    html = response.content

    return html
