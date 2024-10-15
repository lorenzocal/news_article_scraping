'''
Evaluation of the program with the ground truth data
'''
import json
import time

# # should install sentence-transformers
from sentence_transformers import SentenceTransformer, util #SBERT
from sklearn.feature_extraction.text import TfidfVectorizer #TF-IDF
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
# import numpy as np
import matplotlib.pyplot as plt


VECTOR_METHOD = "SBERT"


def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def vectorize_doc(gt_str, own_str) -> (float, float):
    # embedding each doc with SBERT
    if VECTOR_METHOD == 'SBERT':
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model.encode(gt_str), model.encode(own_str)
        
    # embedding each doc with TF-IDF
    elif VECTOR_METHOD == 'TF-IDF':
        vectorizer = TfidfVectorizer()
        gt_vec = vectorizer.fit_transform([gt_str,own_str])[0].toarray()[0]
        own_vec = vectorizer.fit_transform([gt_str,own_str])[1].toarray()[0]
        return gt_vec, own_vec
        
    else:
        raise "Invalid VECTOR_METHOD"


def cos_similarity(gt_str, own_str) -> float:
    gt_vec, own_vec = vectorize_doc(gt_str, own_str)
    return cosine_similarity([gt_vec], [own_vec])[0][0]


def edit_distance(str1, str2):
    m = len(str1)
    n = len(str2)

    # Create a matrix to store results of subproblems
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]

    # Fill dp[][] in bottom up manner
    for i in range(m + 1):
        for j in range(n + 1):
            # If the first string is empty, insert all characters of the second string
            if i == 0:
                dp[i][j] = j    # Insert all characters of str2

            # If the second string is empty, remove all characters of the first string
            elif j == 0:
                dp[i][j] = i    # Remove all characters of str1

            # If last characters of both strings are the same, ignore last character
            # and get the count for the remaining strings
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]

            # If last characters are different, consider all possibilities and
            # find the minimum
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],        # Insert
                                   dp[i - 1][j],        # Remove
                                   dp[i - 1][j - 1])    # Replace

    return dp[m][n]


def normalization_edit_distance(str1, str2):
    """
    Normalization edit distance between two strings
    Args:
    str1: string 1
    str2: string 2

    Returns:
    float: normalized edit distance in 0-1 range
    """
    return edit_distance(str1, str2)/max(len(str1), len(str2))


def calc_edit_distance(gt_str, own_str):
    """
    **Description:**
        - Calculate the normalized edit distance between two list of tokens (the gt text and the retrieved text).
        - The edit distance is normalized by the length of the longest string.
    Args:
        gt_str: ground truth string
        own_str: text extracted from the website

    Returns:
        float: normalized edit distance in 0-1 range
    """
    text_1 = word_tokenize(gt_str)
    text_2 = word_tokenize(own_str)

    val = normalization_edit_distance(text_1, text_2)

    return val

def qualitative_result_switch(x):
    try:
        if x > 0.8:
            return "high"
        elif x > 0.6:
            return "medium"
        elif x > 0.4:
            return "low"
        elif x > 0.2:
            return "very low"
        else:
            return "very very low"
    except:
        return "Error in score value"

def evaluate(gtname, filename):
    # TODO: this function as to be a wrapper for the evaluation metrics
    # TODO: we should decide a qualitative approach to differ the evaluation results in categories:
    #  - 0.8-1. high
    #  -  0.6-0.8 medium
    #  -  0.4-0.6 low
    #  -  0.2-0.4 very low
    #  - 0-0.2 very very low
    # TODO: also decide how and which metrics combine to get the final evaluation result
    """
    **Description:**
        - Evaluate the extracted text with respect to the ground truth text.
        - The evaluation is done using the following methods:
            - Cosine similarity
            - Edit distance
            - Jaccard distance
            - n-grams
            - Longest common substring

    Args:
        gtname: ground truth filename
        filename: text retrieved from the website

    Returns:
        dict: evaluation results
    """
    gt_str = load_data(gtname)
    own_str = load_data(filename)

    evaluation_results = {}

    # first evaluation method is cosine similarity (semantic similarity)
    evaluation_results["cosine_similarity"] = cos_similarity(gt_str, own_str)

    # second evaluation method is edit distance (lexical similarity)
    evaluation_results["normalized_edit_distance"] = calc_edit_distance(gt_str, own_str)

    # third evaluation method is Jaccard
    evaluation_results["jaccard"] = nltk.jaccard_distance(set(word_tokenize(gt_str)), set(word_tokenize(own_str)))

    # fourth evaluation method is n-grams
    # TODO: implement the n-grams method
    evaluation_results["n-grams"] = 0.

    # list of weights for the evaluation methods
    weights = [0.25, 0.25, 0.25, 0.25]

    # do a wheighted average of the first 4 evaluation results
    evaluation_results["evaluation_score"] = weights[0] * evaluation_results["cosine_similarity"]
    + weights[1] * (1 - evaluation_results["normalized_edit_distance"])
    + weights[2] * (1 - evaluation_results["jaccard"])
    + weights[3] * evaluation_results["n-grams"]

    # do a qualitiative evaluation of the evaluation score
    evaluation_results["evaluation_score_qualitative"] = qualitative_result_switch(evaluation_results["evaluation_score"])

    # fifth evaluation method is longest common substring
    evaluation_results["partially_retrieve_metric"] = partially_retrieve_metrics(gt_str, own_str)

    return evaluation_results


def longest_common_substring(str1, str2):
    m = len(str1)
    n = len(str2)

    # Create a 2D array to store lengths of longest common substrings
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    longest = 0  # Length of the longest common substring
    end_index = 0  # Ending index of the longest common substring in str1

    # Build the dp array
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:  # Characters match
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > longest:
                    longest = dp[i][j]
                    end_index = i  # Update end index of LCS
            else:
                dp[i][j] = 0  # Reset if no match

    # Extract the longest common substring from str1 using the end_index
    longest_common_substr = str1[end_index - longest:end_index]

    print("longest_common_substr:", longest_common_substr)


    return longest



def calc_lcsb(gt_str, own_str):
    text_1 = word_tokenize(gt_str)
    text_2 = word_tokenize(own_str)

    val = longest_common_substring(text_1, text_2)

    return val

def test_calc_lcsb():
    gt_str = load_data('data/01.txt')
    own_str = load_data('test_data/01.txt')

    val = calc_lcsb(gt_str, own_str)

    print("LCS:", val)

    percentage = val / len(word_tokenize(gt_str)) * 100

    print("Percentage of LCS:", percentage)

def plot_similarity_edit_distance():
    pre_path = 'scr__/BeautifulSoup_url'

    similarities = []

    print("Edit distance similarity")

    for i in range(1, 6):
        gtname = f'data/0{i}.txt'
        filename = f'{pre_path}{i}.txt'

        # tokenizing the text
        text_1 = word_tokenize(load_data(gtname))
        text_2 = word_tokenize(load_data(filename))

        similarities.append(1 - normalization_edit_distance(text_1, text_2))
        print(f"Similarity between {gtname} and {filename} is {similarities[-1]}")

    # drew an histogram
    plt.bar([f"Pair {i}" for i in range(1, 6)], similarities)
    plt.xlabel("Text Pairs")
    plt.ylabel("(1 - Edit Distance) Similarity")
    plt.title("(1 - Edit Distance) Similarity between Ground Truth and Extracted Texts")

    plt.show()



def all_longest_common_substrings(tokens_list_1, tokens_list_2):
    """
    Find all non-overlapping longest common substrings between two lists of tokens.

    **Idea:**
        - find how many relevant common substrings are there between the two texts.
    Args:
        tokens_list_1: ground truth tokens
        tokens_list_2: retrieved text tokens

    Returns:

    """
    m = len(tokens_list_1)
    n = len(tokens_list_2)

    # Create a 2D array to store lengths of longest common substrings
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # This will store the lengths and ending positions of all common substrings
    substrings = []

    # Build the dp array
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens_list_1[i - 1] == tokens_list_2[j - 1]:  # Tokens match
                dp[i][j] = dp[i - 1][j - 1] + 1
                substrings.append((dp[i][j], i))  # Store length and end position in tokens1
            else:
                dp[i][j] = 0  # Reset if no match

    # Sort substrings by their length in descending order to prioritize longest matches
    substrings.sort(reverse=True, key=lambda x: x[0])

    # To hold the final list of non-overlapping longest common substrings
    non_overlapping_substrings = []
    used_indices = set()

    for length, end_idx in substrings:
        start_idx = end_idx - length

        # Ensure no overlap with already selected substrings
        if all(idx not in used_indices for idx in range(start_idx, end_idx)):
            # Add to the result
            non_overlapping_substrings.append(tokens_list_1[start_idx:end_idx])
            # Mark these indices as used
            used_indices.update(range(start_idx, end_idx))

    return non_overlapping_substrings




def partially_retrieve_metrics(text1, text2):
    """
    **Core Idea:**
        - Find the percentage of ground truth text that is retrieved by the extracted text.

    the idea is to find out the longest common substrings between the ground truth and the extracted text

    keep the one that are longer then a certain threshold of percentage of the length of the ground truth (in terms of token)

    sum the length of all the longest common substrings that are longer than the threshold

    divide the sum by the length of the ground truth
    Returns:
        --- the percentage of the longest common substrings that are longer than the threshold ---
    """
    threshold = 0.05

    text_1 = word_tokenize(text1)
    text_2 = word_tokenize(text2)

    # print("text_1:", text_2)

    lcs_s = all_longest_common_substrings(text_1, text_2)

    # parse and eliminate the ones that are shorter than the threshold
    lcs_s = [lcs for lcs in lcs_s if len(lcs) > threshold * len(text_1)]

    # print("lcs_s:", lcs_s)

    # sum the length of the longest common substrings
    sum_lcs = sum(len(lcs) for lcs in lcs_s)

    # return the percentage of the longest common substrings that are longer than the threshold
    return sum_lcs / len(text_1)
def plot_partially_retrieve_metrics():
    pre_path = 'scr__/BeautifulSoup_url'

    similarities = []

    print("Partially retrieve metrics")

    for i in range(1, 6):
        gtname = f'data/0{i}.txt'
        filename = f'{pre_path}{i}.txt'

        # tokenizing the text
        text_1 = load_data(gtname)
        text_2 = load_data(filename)

        similarities.append(partially_retrieve_metrics(text_1, text_2))
        print(f"Similarity between {gtname} and {filename} is {similarities[-1]}")

    # drew an histogram
    plt.bar([f"Pair {i}" for i in range(1, 6)], similarities)
    plt.xlabel("Text Pairs")
    plt.ylabel("Percentage of retrieved text")
    plt.title("Ratio of the sum of retrieved cs sizes to the gt size")

    plt.show()

