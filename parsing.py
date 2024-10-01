from bs4 import BeautifulSoup


def get_title_and_text(html):
    soup = BeautifulSoup(html.content, 'html.parser')

    title = soup.find('title').get_text()
    
    paragraphs = soup.find_all('p')
    article_text = ' '.join([para.get_text() for para in paragraphs])
    
    return title, article_text