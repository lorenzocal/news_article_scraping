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
    # Download the necessary NLTK resources

    # measure time
    time0 = time.time()
    text_1 = word_tokenize(gt_str)
    text_2 = word_tokenize(own_str)

    print("time taken to tokenize the text: ", time.time()-time0)

    time0 = time.time()
    val = normalization_edit_distance(text_1, text_2)
    print("time taken to calculate edit distance: ", time.time()-time0)

    # assert edit_distance(text_1, text_2) == 13

    # text_2 = word_tokenize(load_data('data/02.txt'))

    return val


def evaluate(gtname, filename, eval_method):
    # TODO: this function as to be a wrapper for the evaluation metrics
    gt_str = load_data(gtname)
    own_str = load_data(filename)

    match eval_method:
        case "cos-sim":
            val = cos_similarity(gt_str, own_str)
        case "edit-dis":
            val = calc_edit_distance(gt_str, own_str)
        case _:
            raise "eval_method not valid"

    print(eval_method, ":", val)

    return val

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
    # measure time
    time0 = time.time()
    text_1 = word_tokenize(gt_str)
    text_2 = word_tokenize(own_str)

    print("time taken to tokenize the text: ", time.time()-time0)

    time0 = time.time()
    val = longest_common_substring(text_1, text_2)
    print("time taken to calculate lcs: ", time.time()-time0)

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



def all_longest_common_substrings(tokens1, tokens2):
    m = len(tokens1)
    n = len(tokens2)

    # Create a 2D array to store lengths of longest common substrings
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # This will store the lengths and ending positions of all common substrings
    substrings = []

    # Build the dp array
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens1[i - 1] == tokens2[j - 1]:  # Tokens match
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
            non_overlapping_substrings.append(tokens1[start_idx:end_idx])
            # Mark these indices as used
            used_indices.update(range(start_idx, end_idx))

    return non_overlapping_substrings




def partially_retrieve_metrics(text1, text2):
    """
    the idea is to ifnd out the longest common substrings between the ground truth and the extracted text

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

