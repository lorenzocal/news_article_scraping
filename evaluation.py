'''
Evaluation of the program with the ground truth data
'''
# should install sentence-transformers
from sentence_transformers import SentenceTransformer, util #SBERT
from sklearn.feature_extraction.text import TfidfVectorizer #TF-IDF
from sklearn.metrics.pairwise import cosine_similarity


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

def evaluate(gtname, filename, vec_method):
    gt_str = load_data(gtname)
    own_str = load_data(filename)

    gt_vec, own_vec = vectorize_doc(gt_str, own_str, vec_method)

    cos_sim = similarity_between_text(gt_vec, own_vec)
    print('cosine similarity :', cos_sim)

    return cos_sim

    
