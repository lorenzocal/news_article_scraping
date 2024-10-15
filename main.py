"""
This is the main file that will be run to test the scraping and parsing functions.

The scraping function will get the HTML of the article from the URL.

The parsing function will extract the title and text of the article from the HTML.

The evaluation function will evaluate the text of the article with respect with the ground truth.
"""
from evaluation import evaluate
from fetching import get_article_simple, get_url_list
import parsing

# url = "https://www.nytimes.com/2024/09/29/us/north-carolina-helene-relief-damage.html"
# url = "https://www.faz.net/aktuell/wirtschaft/kuenstliche-intelligenz/today-s-ai-can-t-be-trusted-19532136.html"

urls = get_url_list()

url = urls[1]

try:
    html = get_article_simple(url)  # step 1
    title, texts = parsing.get_title_and_text1(html)  # step 2
    print("Title:", title)

    print("Text:")
    print(texts)

    #  Evaluate the text
    # TODO: parse and decide how to plot the evaluation
    evaluation = evaluate(title, texts)

    # print("Text:")
    # print(texts)



except Exception as e:
    print(e)
