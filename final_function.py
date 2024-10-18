import pandas as pd
from sklearn.preprocessing import RobustScaler
# Run !pip install sentence_transformers before
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List

from parsing import get_title_and_text1, get_title_and_text2, get_title_and_text3, get_title_and_text4, get_title_and_text5, get_title_and_text6, get_title_and_text7
import cleaning
from fetching import get_article_simple

# Load pre-trained Sentence-BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


# Function to compute coherence within a textual input
def compute_coherence(title: str, text: List[str]) -> float:
    # Title and text are joined and processed together
    text = " ".join([title + "."] + text)

    # Split text into sentences (other techniques may be used for splitting)
    sentences = text.split(".")
    sentences = [sentence.strip() for sentence in sentences if sentence]

    # Get embeddings for each sentence
    embeddings = model.encode(sentences)

    # Compute cosine similarities between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
        similarities.append(similarity)

    # Average similarity score (overall coherence)
    avg_similarity = np.mean(similarities)

    return avg_similarity


def get_optimal_title_text(url: str):
    # List of function defined in the parsing script
    bag_of_functions = [get_title_and_text1, get_title_and_text2, 
                        get_title_and_text3, get_title_and_text4,
                        get_title_and_text5, get_title_and_text6,
                        get_title_and_text7]

    # Two fixed weights are defined for both coherence and abundance
    # We give more importance to the coherence in the text
    ABUNDANCE_WEIGHT = 0.3
    COHERENCE_WEIGHT = 0.7

    # Creating a new dataframe for storing results
    results = pd.DataFrame()

    html = get_article_simple(url)

    for i, function in enumerate(bag_of_functions):
        print("Running function", i)
        title, text = function(html)
        # text = cleaning.clean_text_regressionmatching(title, text)
        coherence = compute_coherence(title, text)
        abundance = len(text)

        # New result row to be added to the dataframe
        new_row = pd.DataFrame([{'title': title, 'text': text, 'coherence': coherence, 'abundance': abundance}])

        # Concatenate the new row with the existing DataFrame
        results = pd.concat([results, new_row], ignore_index=True)

    # Apply Robust Scaling to 'abundance' column
    scaler = RobustScaler()
    results['abundance'] = scaler.fit_transform(results[['abundance']])

    # Assign a rank score to coherence and abundance (the higher the better)
    results['coherence_rank'] = results['coherence'].rank(ascending=True, method='dense').astype(int)
    results['abundance_rank'] = results['abundance'].rank(ascending=True, method='dense').astype(int)

    # The final rank is the weighted average of the two ranks
    results['final_rank'] = COHERENCE_WEIGHT * results['coherence_rank'] + ABUNDANCE_WEIGHT * results['abundance_rank']

    # Get the index of the row with the highest 'final_rank'
    max_rank_index = results['final_rank'].idxmax()

    print("Chosen function:", max_rank_index)
    print(results[['abundance', 'abundance_rank', 'coherence', 'coherence_rank', 'final_rank']])

    best_title = results.loc[max_rank_index, 'title']
    best_text = results.loc[max_rank_index, 'text']

    # Clean the text by semantically similar sentences
    best_text = cleaning.clean_article_semantics(best_title, best_text)

    return best_title, best_text


if __name__ == '__main__':
    url_list = [
        'https://www.nytimes.com/2024/09/29/us/north-carolina-helene-relief-damage.html',
        'http://www.chinatoday.com.cn/ctenglish/2018/commentaries/202409/t20240925_800378506.html',
        'https://english.elpais.com/economy-and-business/2024-10-02/from-failed-inheritances-to-bad-investments-millionaires-who-lost-a-fortune.html'
    ]

    # for each_url in url_list:
    #     title, text = get_optimal_title_text(each_url)
    #     print("-----")
    #     print(f'Title: {title}')
    #     print(f'Text:  {text}')
    #     print('-----')

    title, text = get_optimal_title_text(url_list[0])
    print(f'Title: {title}')
    print(f'Text:  {"".join(text)}')
