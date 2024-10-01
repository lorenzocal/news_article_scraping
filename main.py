"""
This is the main file that will be run to test the scraping and parsing functions.

The scraping function will get the HTML of the article from the URL.

The parsing function will extract the title and text of the article from the HTML.
"""

from scraping import get_article
from parsing import get_title_and_text

url = "https://www.nytimes.com/2024/09/29/us/north-carolina-helene-relief-damage.html"
try:
    html = get_article(url)
    title, text = get_title_and_text(html)
    print("Title:", title)
    print("Text:", text)
except Exception as e:
    print(e)