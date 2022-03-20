import requests
import bs4
import nltk
import numpy as np
header = {
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:32.0) Gecko/20100101 Firefox/32.0', }
base_url = 'https://www.gutenberg.org/'
mywebpage_html = requests.get(
    base_url+'browse/scores/top#books-last30', headers=header, verify=False)
parsed_gutenberg = bs4.BeautifulSoup(mywebpage_html.content, 'html.parser')

# top K books will be downloaded and processed
K = 20


def getpagetext(link):
    print("\nExtracting content for: ", link[0], "Download link: ", link[1])

    page_html = requests.get(link[1], headers=header)
    parsedpage = bs4.BeautifulSoup(page_html.content, 'html.parser')

    # Remove HTML elements that are scripts
    scriptelements = parsedpage.find_all('script')
    # Concatenate the text content from all table cells
    for scriptelement in scriptelements:
        # Extract this script element from the page.
        # This changes the page given to this function!
        scriptelement.extract()
    pagetext = parsedpage.get_text()
    return(pagetext)


def crawler_topK_books(K):
    topK_book_links = parsed_gutenberg.find(
        id="books-last30").findNext('ol').findAll('a')[:K]
    print("Links found: ", len(topK_book_links))
    # link format Baseurl+files+/booknumber+/booknumer-0.txt
    topK_download_links = [
        base_url+"files" + x["href"].replace('/ebooks', '')+x["href"].replace('/ebooks', '')+"-0.txt" for x in topK_book_links]
    topK_books_names = [x.text for x in topK_book_links]
    names_link = zip(topK_books_names, topK_download_links)
    topK_books_content = [getpagetext(link) for link in names_link]

    for name_content in zip(topK_books_names, topK_books_content):

        with open('./'+name_content[0].replace(':', '').replace('?', '')+".txt", "w", encoding="utf-8") as oFile:

            oFile.write(name_content[1])
            oFile.close()

    return list(map(list, zip(topK_books_names, topK_books_content)))


def tagtowordnet(postag):
    wordnettag = -1
    if postag[0] == 'N':
        wordnettag = 'n'
    elif postag[0] == 'V':
        wordnettag = 'v'
    elif postag[0] == 'J':
        wordnettag = 'a'
    elif postag[0] == 'R':
        wordnettag = 'r'
    return(wordnettag)


lemmatizer = nltk.stem.WordNetLemmatizer()


def lemmatizetext(nltktexttolemmatize):
    # Tag the text with POS tags
    taggedtext = nltk.pos_tag(nltktexttolemmatize)
    # Lemmatize each word text
    lemmatizedtext = []
    for l in range(len(taggedtext)):
        # Lemmatize a word using the WordNet converted POS tag
        wordtolemmatize = taggedtext[l][0]
        wordnettag = tagtowordnet(taggedtext[l][1])
        if wordnettag != -1:
            lemmatizedword = lemmatizer.lemmatize(wordtolemmatize, wordnettag)
        else:
            lemmatizedword = wordtolemmatize
    # Store the lemmatized word
        lemmatizedtext.append(lemmatizedword)
    return(lemmatizedtext)


names_content = crawler_topK_books(K)

# text cleanup
for idx, items in enumerate(names_content):
    items[1] = items[1].strip()
    items[1] = ' '.join(items[1].split())
    names_content[idx] = items


# tokenization and nltk formatting
nltk_text = []
for items in names_content:
    tokenized_content = nltk.word_tokenize(items[1])
    # lowercase after tokenizing
    tokenized_content = map(lambda x: x.lower(), tokenized_content)
    nltk_text.append(nltk.Text(tokenized_content))

# converting lematized text to nltk Text object
nltk_text_lematized = []
for text in nltk_text:
    nltk_text_lematized.append(nltk.Text(lemmatizetext(text)))


# Combining unique vocablist to form a unified vocab list and find top 100 words from Total
# Corpus of all text
vocab = []
vocab_index = []
total_corpus = []
for lematized_text in nltk_text_lematized:

    unique_words = np.unique(lematized_text, return_inverse=True)
    vocab.append(unique_words[0])
    vocab_index.append(unique_words[1])
    total_corpus.extend(lematized_text)

corpus_vocab_unique, counts = np.unique(total_corpus, return_counts=True)
sorted_index = sorted(range(len(counts)),
                      key=lambda k: counts[k], reverse=True)
corpus_vocab_unique = corpus_vocab_unique[sorted_index]
counts = counts[sorted_index]


print("\n\nUnified Vocab length: ", len(corpus_vocab_unique))
print("\nTop 100 words in all documents")
print(corpus_vocab_unique[:100])

#Testing githubs
