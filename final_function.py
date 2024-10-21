import pandas as pd
from sklearn.preprocessing import RobustScaler
# Run !pip install sentence_transformers before
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

from parsing import get_title_and_text1, get_title_and_text2, get_title_and_text3, get_title_and_text4, get_title_and_text5, get_title_and_text6, get_title_and_text7
import cleaning
from fetching import get_article_simple, get_url_list

# Load pre-trained Sentence-BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


# Function to compute coherence within a textual input
def compute_coherence(title: str, text: list[str]) -> float:
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
        try : 
            title, text = function(html)
            # text = cleaning.clean_text_regressionmatching(title, text)
            coherence = compute_coherence(title, text)
            abundance = len(text)

            # New result row to be added to the dataframe
            new_row = pd.DataFrame([{'title': title, 'text': text, 'coherence': coherence, 'abundance': abundance}])

            # Concatenate the new row with the existing DataFrame
            results = pd.concat([results, new_row], ignore_index=True)
        except Exception as e:
            print(f"Function {i} failed.")

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
    # print(results[['abundance', 'abundance_rank', 'coherence', 'coherence_rank', 'final_rank']])

    best_title = results.loc[max_rank_index, 'title']
    best_text = results.loc[max_rank_index, 'text']

    # Clean the text by semantically similar sentences
    best_text = cleaning.clean_article_semantics(best_title, best_text)

    return best_title, best_text


if __name__ == '__main__':
    # Fetch all website from the URL list
    # Save the generated title and text to a file
    # Save as one big JSON file (that might not exist), in /data/ of the form:
    # {
    #     "url1": {
    #         "title": "Best title",
    #         "text": ["Best sentence 1", "Best sentence 2", ...]
    #     },
    #     "url2": {
    #         "title": "Best title",
    #         "text": ["Best sentence 1", "Best sentence 2", ...]
    #     },
    #     ...
    # }
    url_list = get_url_list()
    results = {}
    for url in url_list:
        print("Processing URL:", url)
        title, text = get_optimal_title_text(url)
        results[url] = {'title': title, 'text': text}

    with open('./data/extracted_data.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Data saved to extracted_data.json")