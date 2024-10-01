# news_article_scraping

Web scraper to try to retrive as much as possible text from different newspapers.
Developped as part of a course on Natural Language Processing.

## Subject

News article scraper.

Write a program or notebook that downloads a journal article and extracts its title and its text.

The program takes as input the url of the news article. It produces as output the title and the text of the article. You can (but you don't have to) follow the following steps.

Write a function that downloads the html page associated to the news article (input parameter : the url of the article, output : the html source code of the article.)
Write a method that takes as input a text containing an html page of a news article and extracts the title and the text from that page. Check the following libraries
https://newspaper.readthedocs.io/en/latest/
https://textract.readthedocs.io/en/stable/
https://goose3.readthedocs.io/en/latest/index.html
https://github.com/kotartemiy/newscatcher
https://github.com/alan-turing-institute/ReadabiliPy
search for other libraries


Test your code with the following urls:
https://www.nytimes.com/2024/09/29/us/north-carolina-helene-relief-damage.html
https://www.faz.net/aktuell/wirtschaft/kuenstliche-intelligenz/today-s-ai-can-t-be-trusted-19532136.html
http://www.chinatoday.com.cn/ctenglish/2018/commentaries/202409/t20240925_800378506.html
https://english.elpais.com/economy-and-business/2024-09-28/from-the-hermes-heir-to-nicolas-cage-millionaires-who-went-bankrupt.html
https://insatiable.info/2023/06/30/quels-futur-pour-les-reseaux-sociaux/
https://actu.fr/auvergne-rhone-alpes/lyon_69123/lyon-le-projet-de-reamenagement-des-quais-les-plus-mortels-pour-les-cyclistes-devoile_61667371.html


Propose a metric to evaluate the accuracy of your title and text extraction method.