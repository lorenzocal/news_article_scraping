from bs4 import BeautifulSoup

# get the title of the HTML
def get_title_and_text(html: bytes) -> (str, list[str]):
    soup = BeautifulSoup(html, 'html.parser')

    title = soup.find('title').get_text()
    
    # paragraphs = soup.find_all('p')

    # article_text = ' '.join([para.get_text() for para in paragraphs])
    strings = soup.body.strings # Get all strings from within the HTML
    
    return title, strings
