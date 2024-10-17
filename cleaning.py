import spacy

# Convert list of strings to article
def los_to_article(list_ : list[str]) -> str:
    return '\n'.join(list_)

def clean_article_semantics(title : str, paragraphs : list[str]) -> str:
    """
    Get the title and text of the HTML using semantic similarity
    Args:
        title: The title of the article
        paragraphs: The paragraphs of the article as a list of strings
    Returns:
        str: The cleaned article
    """

    nlp = spacy.load("en_core_web_lg")

    article_so_far = [title]
    nlp_article_so_far = nlp(los_to_article(article_so_far))

    # Remove the title from the paragraphs
    paragraphs.remove(title)

    # Remove paragraphs that are too short
    paragraphs = [paragraph for paragraph in paragraphs if len(paragraph) > 10]

    # Iteratively add paragraphs to the article until the similarity between the article and the new paragraph is less than 0.9
    banned_paragraphs = []
    for paragraph in paragraphs:
        nlp_paragraph = nlp(paragraph)
        similarity = nlp_article_so_far.similarity(nlp_paragraph)
        # print(f"{similarity} : {paragraph}")
        if similarity > 0.55:
            article_so_far += [paragraph]
            nlp_article_so_far = nlp(los_to_article(article_so_far))
        else:
            banned_paragraphs.append(paragraph)

    article_so_far.remove(title)

    # Go through the article and remove paragraphs that are not similar enough with the rest of the article
    banned_paragraphs = []
    for paragraph in article_so_far:
        nlp_paragraph = nlp(paragraph)
        if nlp_article_so_far.similarity(nlp_paragraph) < 0.8:
            banned_paragraphs.append(paragraph)
    
    for paragraph in banned_paragraphs:
        article_so_far.remove(paragraph)

    return los_to_article(article_so_far)

def clean_text_regressionmatching(title: str, paragraphs: list[str]) -> str:
    """
    Remove advertisements and other unwanted text from the article's paragraphs
    in multiple languages (English, French, German, Italian), and also remove
    paragraphs that are too short.
    """

    # Define the unwanted phrases for removal (in multiple languages)
    unwanted_phrases = [
        'Advertisement',     # English
        'Supported by',      # English
        'publicité',         # French
        'soutenu par',       # French
        'Werbung',           # German
        'unterstützt von',   # German
        'pubblicità',        # Italian
        'supportato da'      # Italian
    ]

    # Filter out paragraphs containing any of the unwanted phrases
    filtered_paragraphs = [
        paragraph for paragraph in paragraphs
        if not any(phrase in paragraph for phrase in unwanted_phrases)
    ]

    # Remove paragraphs that are too short (length <= 10 characters)
    filtered_paragraphs = [
        paragraph for paragraph in filtered_paragraphs if len(paragraph) > 10
    ]

    # Combine the cleaned paragraphs with the title and return as a single string
    return title + '\n' + '\n'.join(filtered_paragraphs)

    
    

# Other idea
# Iteratively add paragraphs to the article until the similarity between the article and the article with the new paragraph is less than 0.9
