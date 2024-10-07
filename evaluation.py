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



def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def vectorize_doc(gt_str, own_str, vector_method):
    # embedding each doc with SBERT
    if vector_method == 'SBERT':
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model.encode(gt_str), model.encode(own_str)
        
    # embedding each doc with TF-IDF
    elif vector_method == 'TF-IDF':
        vectorizer = TfidfVectorizer()
        gt_vec = vectorizer.fit_transform([gt_str,own_str])[0].toarray()[0]
        own_vec = vectorizer.fit_transform([gt_str,own_str])[1].toarray()[0]
        return gt_vec, own_vec
        
    else:
        return None, None

def similarity_between_text(gt_vec,own_vec):
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


def evaluate(gtname, filename, vec_method):
    gt_str = load_data(gtname)
    own_str = load_data(filename)

    gt_vec, own_vec = vectorize_doc(gt_str, own_str, vec_method)

    cos_sim = similarity_between_text(gt_vec, own_vec)
    print('cosine similarity :', cos_sim)

    return cos_sim


def test_edit_distance():
    # Download the necessary NLTK resources
    #

    # measure time
    time0 = time.time()
    text_1 = word_tokenize(load_data('data/01.txt'))
    text_2 = word_tokenize(load_data('test_data/01.txt'))

    print("time taken to tokenize the text: ", time.time()-time0)

    time0 = time.time()
    print(normalization_edit_distance(text_1, text_2))
    print("time taken to calculate edit distance: ", time.time()-time0)

    assert edit_distance(text_1, text_2) == 13

    text_2 = word_tokenize(load_data('data/02.txt'))

    time0 = time.time()
    print(normalization_edit_distance(text_1, text_2))
    print("time taken to calculate edit distance: ", time.time()-time0)

