"""
This is the main file that will be run to test the scraping and parsing functions.

The scraping function will get the HTML of the article from the URL.

The parsing function will extract the title and text of the article from the HTML.
"""

from scraping import get_article_simple, get_url_list
from parsing import get_title_and_text, get_title_and_text2, get_title_and_text3, get_title_and_text4

# url = "https://www.nytimes.com/2024/09/29/us/north-carolina-helene-relief-damage.html"
# url = "https://www.faz.net/aktuell/wirtschaft/kuenstliche-intelligenz/today-s-ai-can-t-be-trusted-19532136.html"

urls = get_url_list()

try:
    html = get_article_simple(urls[1])  # step 1
    title, texts = get_title_and_text(html)  # step 2
    print("Title:", title)

    print("Text:")
    print(texts)

except Exception as e:
    print(e)
