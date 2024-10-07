'''
Evaluation of the program with the ground truth data
'''
import time

# # should install sentence-transformers
from sentence_transformers import SentenceTransformer, util #SBERT
from sklearn.feature_extraction.text import TfidfVectorizer #TF-IDF
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize


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