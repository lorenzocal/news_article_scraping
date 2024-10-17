from fetching import get_article_simple, get_url_list
import cleaning
from bs4 import BeautifulSoup
import html

urls = get_url_list()

# Get from new york times
html_content = get_article_simple(urls[2])

# Convert the string to bytes
html_content = html_content.decode('utf-8')
html_content = html.unescape(html_content)
soup = BeautifulSoup(html_content, 'html.parser')

# Get title 
title = soup.find('title').get_text()

# Get all the text from the HTML split into paragraphs
paragraphs = soup.get_text(separator='\n', strip=True).split('\n')

article = cleaning.clean_article_semantics(title, paragraphs)
print("Title:", title)
# print("Text:", article)

print("Finish")