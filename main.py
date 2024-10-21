"""
This is the main file that will be run to test the scraping and parsing functions.

The scraping function will get the HTML of the article from the URL.

The parsing function will extract the title and text of the article from the HTML.

The evaluation function will evaluate the text of the article with respect with the ground truth.
"""
import os

from evaluation import evaluate
from fetching import get_article_simple, get_url_list
import parsing

import evaluation
import fetching
import final_function
import json


def print_retrieve_text(url_index, title, texts):
    print("Title:", title)
    print("Text:")
    print(texts)
    name_file = "retrieved_article{}.txt".format(url_index)

    file_path = "./retrieved_articles/" + name_file

    # Ensure the directory exists
    directory = "./retrieved_articles/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(f"{title}\n")
        file.write(f"{texts}")

    return file_path


def scrape_text_and_save(url_index):
    urls = fetching.get_url_list()
    url = urls[url_index]

    # use the new optimised function
    title, texts = final_function.get_optimal_title_text(url)  # selects the best parsing function, then cleans

    # html = fetching.get_article_simple(url)  # step 1
    # title, texts = parsing.get_title_and_text1(html)  # step 2

    # print then store the text to a file, so it is easier to read
    return print_retrieve_text(url_index, title, texts)

def print_evaluation_in_json(evaluation, index):
    # construct the dictionary as
    url_name = ""
    url_name = url_name + "{}".format(get_url_list()[index])

    result = {
        url_name: evaluation
    }

    # append the dictionary in the json file
    with open("./evaluations/evaluations.json", "a") as file:
        json.dump(result, file, indent=4)

index = 1
scraped_path = scrape_text_and_save(index)

#  Evaluate the text
# TODO: parse and decide how to plot the evaluation
gt_path = "./data/GroundTruth/0{}.txt".format(index+1)
evaluation = evaluation.evaluate(gt_path, scraped_path)

print("Evaluation:")
print(evaluation)

print_evaluation_in_json(evaluation, index)
