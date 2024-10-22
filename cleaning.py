import spacy
import re
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

    # Step 2: Define unwanted phrases (advertisements) in multiple languages using regular expressions
    # Also includes patterns for URLs and social media links in multiple languages
    unwanted_phrases = [
        # English phrases
        r'advertisement',                    # English
        r'supported by',                     # English
        r'follow us on',                     # English social media prompt
        r'click here',                       # English call-to-action
        r'subscribe to our',                 # English subscription-related spam
        r'visit our website',                # English promotional phrase
        r'order now',                        # English ordering call-to-action
        r'discount',                         # English discount-related phrases
    
        # French phrases
        r'publicit[eé]',                     # French (advertisement with accent variations)
        r'soutenu par',                      # French (supported by)
        r'suivez[-\s]nous sur',              # French (follow us on)
        r'cliquez ici',                      # French (click here)
        r'abonnez[-\s]vous',                 # French (subscribe to)
        r'visitez notre site',               # French (visit our website)
        r'commandez[-\s]maintenant',         # French (order now)
        r'r[eé]duction',                     # French (discount)
    
        # German phrases
        r'werbung',                          # German (advertisement)
        r'unterst[üu]tzt von',               # German (supported by, with ü/u variations)
        r'folgen sie uns auf',               # German (follow us on)
        r'klicken sie hier',                 # German (click here)
        r'abonnieren sie',                   # German (subscribe to)
        r'besuchen sie unsere webseite',     # German (visit our website)
        r'jetzt bestellen',                  # German (order now)
        r'rabatt',                           # German (discount)
    
        # Italian phrases
        r'pubblicit[àa]',                    # Italian (advertisement with accent variations)
        r'supportato da',                    # Italian (supported by)
        r'seguici su',                       # Italian (follow us on)
        r'clicca qui',                       # Italian (click here)
        r'abbonati a',                       # Italian (subscribe to)
        r'visita il nostro sito',            # Italian (visit our website)
        r'ordina ora',                       # Italian (order now)
        r'sconto',                           # Italian (discount)
    
        # URL patterns (common in advertisements)
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',  # Generic URL pattern
        r'www\.[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}', # "www." links
    
        # Social media or generic spammy phrases often seen in ads (multilingual variants)
        r'suivez[-\s]nous sur',              # Multilingual: follow us on (common in French and others)
        r'seguici su',                       # Multilingual: follow us on (Italian)
        r'folgen sie uns auf',               # German: follow us on
    ]

    # Step 3: Remove paragraphs containing unwanted phrases using regular expressions
    filtered_paragraphs = [
        paragraph for paragraph in paragraphs
        if not any(re.search(phrase, paragraph, re.IGNORECASE) for phrase in unwanted_phrases)
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
