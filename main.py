"""
This is the main file that will be run to test the scraping and parsing functions.

The scraping function will get the HTML of the article from the URL.

The parsing function will extract the title and text of the article from the HTML.

The evaluation function will evaluate the text of the article with respect with the ground truth.
"""
import os

import matplotlib.pyplot as plt
from fetching import get_url_list
import evaluation
import fetching
import final_function
import json


def print_and_save_retrieved_text(url_index, title, texts):
    print("Title:", title)
    print("Text:")
    print(texts)
    name_file = f"retrieved_article{url_index}.txt"

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
    return print_and_save_retrieved_text(url_index, title, texts)


def print_evaluation_in_json(evaluation, index):
    # construct the dictionary as
    url_name = ""
    url_name = url_name + "{}".format(get_url_list()[index])

    result = {
        url_name: evaluation
    }

    directory = "./evaluations"
    file_path = "./evaluations/evaluations.json"

    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            data["evaluations"].append(result)

        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)
    except FileNotFoundError:
        with open(file_path, "w") as file:
            json.dump({"evaluations": [result]}, file, indent=4)

def save_graph_evaluation(evaluation, index):

    directory = "./eval_graphs"
    if not os.path.exists(directory):
        os.makedirs(directory)

    x = ['cos_sim', 'edit_d', 'jaccard', 'ngrams', 'total', 'part']
    y = list(evaluation.values())
    y_quant = y[5]
    y.pop(5)

    plt.bar(x, y)
    plt.title(f"article_no.{index}")
    plt.xlabel("evaluation metrics")
    plt.ylabel("normalized similarities")
    plt.ylim(0, 1.1)
    plt.grid(axis='y')
    for j, value in enumerate(y):
        plt.text(j, value, f'{value:.3f}', ha='center', va='bottom')

    plt.savefig(f"{directory}/article_no_{index}.png")    
    plt.clf()
    plt.close()

def save_metrics_graph(alls):
    metrics = ['cos_sim', 'edit_d', 'jaccard', 'ngrams', 'total', 'part']
    for k in range(len(metrics)):
        directory = "./eval_graphs_metrics"
        if not os.path.exists(directory):
            os.makedirs(directory)

        articles = []
        for l in range(len(fetching.get_url_list())):
            articles.append(all_evals[l][k])

        x = ['article1', 'article2', 'article3', 'article4', 'article5', 'article6']
        y = articles

        plt.bar(x, y)
        plt.title(f"metric = {metrics[k]}")
        plt.xlabel("articles")
        plt.ylabel(f"{metrics[k]}_similarities")
        plt.ylim(0, 1.1)
        plt.grid(axis='y')
        for j, value in enumerate(y):
            plt.text(j, value, f'{value:.3f}', ha='center', va='bottom')

        plt.savefig(f"{directory}/metric_{metrics[k]}.png")    
        plt.clf()
        plt.close()


all_evals = []
for i in range(len(fetching.get_url_list())):
    print(f"article_{i} starts")
    scraped_path = scrape_text_and_save(i)

    #  Evaluate the text
    # TODO: parse and decide how to plot the evaluation
    gt_path = "./data/GroundTruth/0{}.txt".format(i+1)
    final_eval = evaluation.evaluate(gt_path, scraped_path)

    print("Evaluation:")
    print(final_eval)

    print_evaluation_in_json(final_eval, i)
    save_graph_evaluation(final_eval, i)

    evals2 = list(final_eval.values())
    evals2.pop(5)
    all_evals.append(evals2)
    print('====================================\n\n')

save_metrics_graph(all_evals)


# =======================================================================================================================
# GET ALL ARTICLES AND SAVE THEM TO TEXT
# url_list = get_url_list()
#
# for i, url in enumerate(url_list) :
#     print(f"Processing URL {i}: {url}")
#     title, text = final_function.get_optimal_title_text(url)
#     print_and_save_retrieved_text(i, title, text)
