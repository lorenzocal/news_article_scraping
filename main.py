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

def print_retrieve_text(url_index, title, texts):
    print("Title:", title)
    print("Text:")
    print(texts)
    name_file = "retrive_article{}.txt".format(url_index)

    file_path = "./retrieve_articles/" + name_file

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(f"{title}\n")
        file.write(f"{texts}")

    return file_path


url_index = 1

urls = get_url_list()

url = urls[url_index]

try:
    html = get_article_simple(url)  # step 1
    title, texts = parsing.get_title_and_text1(html)  # step 2
    print("Title:", title)

    print("Text:")
    print(texts)

    # print the text to a file so is easier to read it
    file_path_retr = print_retrieve_text(url_index, title, texts)

    #  Evaluate the text
    # TODO: parse and decide how to plot the evaluation
    gt_path = "./data/GroundTruth/0{}.txt".format(url_index)
    evaluation = evaluate(gt_path, file_path_retr)
    print("Evaluation:")
    print(evaluation)

    # print("Text:")
    # print(texts)



except Exception as e:
    print(e)
