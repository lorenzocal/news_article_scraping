import spacy

# Combined function that cleans the article by removing unwanted paragraphs and 
# filtering the content based on semantic similarity

def clean_article_semantics(title: str, paragraphs: list[str]) -> str:
    """
    Clean the article by first removing unwanted advertisements or phrases in multiple languages,
    and then applying semantic similarity filtering to retain only coherent paragraphs.

    Args:
        title: The title of the article.
        paragraphs: The paragraphs of the article as a list of strings.

    Returns:
        str: The final cleaned article.
    """

    nlp = spacy.load("en_core_web_lg")

    # Convert list of paragraphs to a single article string
    def los_to_article(list_: list[str]) -> str:
        return '\n'.join(list_)

    # Step 1: Remove the title from the paragraphs if present
    if title in paragraphs:
        paragraphs.remove(title)

    # Step 2: Define unwanted phrases (advertisements) in multiple languages
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

    # Step 3: Remove paragraphs containing unwanted phrases
    filtered_paragraphs = [
        paragraph for paragraph in paragraphs
        if not any(phrase in paragraph for phrase in unwanted_phrases)
    ]

    # Step 4: Remove paragraphs that are too short (length <= 10 characters)
    filtered_paragraphs = [
        paragraph for paragraph in filtered_paragraphs if len(paragraph) > 10
    ]

    # Step 5: Semantic similarity cleaning process
    # Start with the title as the initial "article so far"
    article_so_far = [title]
    nlp_article_so_far = nlp(los_to_article(article_so_far))

    # Step 6: Iteratively add paragraphs based on semantic similarity
    banned_paragraphs = []
    for paragraph in filtered_paragraphs:
        nlp_paragraph = nlp(paragraph)
        similarity = nlp_article_so_far.similarity(nlp_paragraph)
        if similarity > 0.55:
            # Add paragraph to article if similarity is above the threshold
            article_so_far += [paragraph]
            nlp_article_so_far = nlp(los_to_article(article_so_far))
        else:
            # If not similar enough, ban the paragraph
            banned_paragraphs.append(paragraph)

    # Remove the title from the final article content (keep it separate)
    article_so_far.remove(title)

    # Step 7: Further filter the article by removing paragraphs that are not similar
    final_banned_paragraphs = []
    for paragraph in article_so_far:
        nlp_paragraph = nlp(paragraph)
        # If similarity is less than 0.8, remove the paragraph
        if nlp_article_so_far.similarity(nlp_paragraph) < 0.8:
            final_banned_paragraphs.append(paragraph)
    
    # Remove paragraphs that were flagged as not sufficiently similar
    for paragraph in final_banned_paragraphs:
        article_so_far.remove(paragraph)

    # Return the final cleaned article content as a list of paragraphs
    return article_so_far
