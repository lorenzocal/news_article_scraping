"""
Evaluation of the program with the ground truth data
"""
from collections import Counter
import time
import nltk
# # should install sentence-transformers
from sentence_transformers import SentenceTransformer  # SBERT
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk import jaccard_distance, ngrams
# import numpy as np
import matplotlib.pyplot as plt


VECTOR_METHOD = "TF-IDF"
NGRAM_N = 2


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
        gt_vec = vectorizer.fit_transform([gt_str, own_str])[0].toarray()[0]
        own_vec = vectorizer.fit_transform([gt_str, own_str])[1].toarray()[0]
        return gt_vec, own_vec
        
    else:
        raise "Invalid VECTOR_METHOD"

def qualitative_result_switch(x: float):
    try:
        if x > 0.9:
            return "very high"
        elif x > 0.8:
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

# Cosine Similarity function
def cos_similarity(gt_str, own_str) -> float:
    start_time = time.time()  # Start time

    gt_vec, own_vec = vectorize_doc(gt_str, own_str)
    similarity = cosine_similarity([gt_vec], [own_vec])[0][0]

    end_time = time.time()  # End time
    elapsed_time = end_time - start_time

    # Write time to a file
    with open("cos_similarity_time.txt", "a") as file:
        file.write(f"Time taken: {elapsed_time} seconds\n")

    return similarity

# Edit Distance function
def edit_distance(str1, str2):
    start_time = time.time()  # Start time

    m = len(str1)
    n = len(str2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])

    end_time = time.time()  # End time
    elapsed_time = end_time - start_time

    # Write time to a file
    with open("edit_distance_time.txt", "a") as file:
        file.write(f"Time taken: {elapsed_time} seconds\n")

    return dp[m][n]

# Normalized Edit Distance
def edit_distance_normalized(str1, str2):
    start_time = time.time()  # Start time

    distance = edit_distance(str1, str2)
    normalized_distance = distance / max(len(str1), len(str2))

    end_time = time.time()  # End time
    elapsed_time = end_time - start_time

    # Write time to a file
    with open("edit_distance_normalized_time.txt", "a") as file:
        file.write(f"Time taken: {elapsed_time} seconds\n")

    return normalized_distance

# Calculate Normalized Edit Distance
def calc_edit_distance(gt_str, own_str):
    start_time = time.time()  # Start time

    text_1 = word_tokenize(gt_str)
    text_2 = word_tokenize(own_str)
    val = edit_distance_normalized(text_1, text_2)

    end_time = time.time()  # End time
    elapsed_time = end_time - start_time

    # Write time to a file
    with open("calc_edit_distance_time.txt", "a") as file:
        file.write(f"Time taken: {elapsed_time} seconds\n")

    return val

# Longest Common Substring
def longest_common_substring(str1, str2):
    start_time = time.time()  # Start time

    m = len(str1)
    n = len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    longest = 0
    end_index = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > longest:
                    longest = dp[i][j]
                    end_index = i
            else:
                dp[i][j] = 0

    end_time = time.time()  # End time
    elapsed_time = end_time - start_time

    # Write time to a file
    with open("longest_common_substring_time.txt", "a") as file:
        file.write(f"Time taken: {elapsed_time} seconds\n")

    return longest

# Calculate Longest Common Substring
def calc_lcsb(gt_str, own_str):
    start_time = time.time()  # Start time

    text_1 = word_tokenize(gt_str)
    text_2 = word_tokenize(own_str)
    val = longest_common_substring(text_1, text_2)

    end_time = time.time()  # End time
    elapsed_time = end_time - start_time

    # Write time to a file
    with open("calc_lcsb_time.txt", "a") as file:
        file.write(f"Time taken: {elapsed_time} seconds\n")

    return val

# Find all non-overlapping longest common substrings between two lists of tokens
def all_longest_common_substrings(tokens_list_1, tokens_list_2):
    m = len(tokens_list_1)
    n = len(tokens_list_2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    substrings = []

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens_list_1[i - 1] == tokens_list_2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                substrings.append((dp[i][j], i))
            else:
                dp[i][j] = 0

    substrings.sort(reverse=True, key=lambda x: x[0])
    non_overlapping_substrings = []
    used_indices = set()

    for length, end_idx in substrings:
        start_idx = end_idx - length
        if all(idx not in used_indices for idx in range(start_idx, end_idx)):
            string_to_add = tokens_list_1[start_idx:end_idx]
            if string_to_add not in non_overlapping_substrings and len(string_to_add) > 3:
                non_overlapping_substrings.append(string_to_add)
            used_indices.update(iter(range(start_idx, end_idx)))

    return non_overlapping_substrings

# Partially retrieve metrics
def partially_retrieve_metrics(text1: str, text2: str):
    start_time = time.time()  # Start time

    text_1 = word_tokenize(text1)
    text_2 = word_tokenize(text2)
    lcs_s = all_longest_common_substrings(text_1, text_2)
    lcs_s = [lcs for lcs in lcs_s if len(lcs) > 0.01 * len(text_1)]
    sum_lcs = sum(len(lcs) for lcs in lcs_s)
    percentage = sum_lcs / len(text_1)

    end_time = time.time()  # End time
    elapsed_time = end_time - start_time

    # Write time to a file
    with open("partially_retrieve_metrics_time.txt", "a") as file:
        file.write(f"Time taken: {elapsed_time} seconds\n")

    return percentage


def n_grams(text1: str, text2: str, n: int) -> float:
    # Start time recording
    start_time = time.time()

    # Tokenize and lowercase text
    tokens1 = word_tokenize(text1.lower())
    tokens2 = word_tokenize(text2.lower())
    ngrams1 = list(ngrams(tokens1, n))
    ngrams2 = list(ngrams(tokens2, n))

    # Count frequency of each n-gram in both texts
    counter1 = Counter(ngrams1)
    counter2 = Counter(ngrams2)

    # Find common n-grams between both texts
    common_ngrams = counter1 & counter2  # Intersection: min of counts

    # Calculate the total number of n-grams for similarity
    total_ngrams = len(ngrams1) + len(ngrams2)
    common_count = sum(common_ngrams.values())

    # Compute similarity score
    similarity = (2 * common_count) / total_ngrams if total_ngrams > 0 else 0

    # End time recording
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Write the elapsed time to a file
    with open("execution_time_n_gram.txt", "a") as file:
        file.write(f"Time taken for n-gram similarity: {elapsed_time} seconds\n")

    return similarity


def evaluate(gtname, filename):
    # TODO: this function as to be a wrapper for the evaluation metrics
    # TODO: we should decide a qualitative approach to differ the evaluation results in categories:
    #  - 0.8-1. high
    #  - 0.6-0.8 medium
    #  - 0.4-0.6 low
    #  - 0.2-0.4 very low
    #  - 0-0.2 very very low
    # TODO: also decide how and which metrics combine to get the final evaluation result
    """
    Evaluates extracted text against the ground truth text using multiple methods.

    This function compares the text retrieved from the website to the ground truth
    text by employing several evaluation techniques, including:

        - Cosine similarity (semantic similarity)
        - Edit distance (lexical similarity)
        - Jaccard distance
        - N-grams similarity

    Args:
        gtname (str): The filename containing the ground truth text.
        filename (str): The filename containing the text retrieved from the website.

    Returns:
        A dictionary containing the evaluation results from each method, as well as a generic evaluation score.
        {
            "cosine_similarity": float,
            "normalized_edit_distance": float,
            "jaccard": float,
            "n-grams": float,
            "evaluation_score": float,
            "evaluation_score_qualitative": str,
            "partially_retrieve_metric": float
        }
    """

    nltk.download('punkt_tab')

    gt_str = load_data(gtname)
    own_str = load_data(filename)

    evaluation_results = {
        # first evaluation method is cosine similarity (semantic similarity)
        "cosine_similarity": cos_similarity(gt_str, own_str), # just for testing

        # second evaluation method is edit distance (lexical similarity)
        "1 - normalized_edit_distance": 1-calc_edit_distance(gt_str, own_str),

        # third evaluation method is Jaccard
        "1 - jaccard": 1-jaccard_distance(set(word_tokenize(gt_str)), set(word_tokenize(own_str))),

        # fourth evaluation method is n-grams
        # TODO: implement the n-grams method
        "n-grams": n_grams(gt_str, own_str, NGRAM_N)  # just for testing
    }

    # list of weights for the evaluation methods
    weights = [0.15, 0.35, 0.15, 0.35]

    # do a weighted average of the first 4 evaluation results
    evaluation_results["evaluation_score"] = (weights[0] * evaluation_results["cosine_similarity"]
                                              + weights[1] * (evaluation_results["1 - normalized_edit_distance"])
                                              + weights[2] * (evaluation_results["1 - jaccard"])
                                              + weights[3] * evaluation_results["n-grams"])

    # do a qualitative evaluation of the evaluation score
    evaluation_results["evaluation_score_qualitative"] = qualitative_result_switch(evaluation_results["evaluation_score"])

    # fifth evaluation method is the longest common substring
    evaluation_results["partially_retrieve_metric"] = partially_retrieve_metrics(gt_str, own_str)

    # round all the evaluation score to 4 decimal places
    evaluation_results = {key: round(value, 4) if isinstance(value, float) else value for key, value in evaluation_results.items()}

    return evaluation_results

def calculate_average_times(file_names):
    result_dict = {}
    for file_name in file_names:
        total_time = 0
        count = 0

        with open(file_name, 'r') as file:
            for line in file:
                if "Time taken: " in line:
                    try:
                        time_taken = float(line.split("Time taken: ")[1].split(" seconds")[0])
                        total_time += time_taken
                        count += 1
                    except (IndexError, ValueError):
                        print(f"Warning: Could not parse time in line: {line.strip()}")
                        continue
        
        average_time = total_time / count if count > 0 else 0
        result_dict[file_name] = average_time

    return result_dict



# Example usage
file_names = ["cos_similarity_time.txt","calc_edit_distance_time.txt","edit_distance_time.txt","execution_time_n_gram.txt","partially_retrieve_metrics_time.txt"]
# Dictionary to store average times for each file
average_time_dict = calculate_average_times(file_names)
filename_labels = [
    "cos_similarity",
    "calc_edit_distance",
    "edit_distance",
    "n_gram",
    "partially_retrieve_metrics"
]


average_times = list(average_time_dict.values())
plt.figure(figsize=(16, 12))
plt.barh(filename_labels, average_times, color='skyblue')
plt.xlabel('Average Time (s)')
plt.ylabel('File Names')
plt.title('Average Times for Different Files')
plt.savefig('average_times_chart.png')



# Output the result
# print(f"Average time for {file_name}: {average_time_dict[file_name]} seconds")
