"""
Topic extraction from tweets with Latent Dirichlet Allocation (LDA).
"""
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
nlp = spacy.load('en_core_web_sm')


file = 'ewarren'
regex = r'https?:\/\/.*[\r\n]*|\@[A-Za-z0-9]*|\#[A-Za-z0-9]*|pic.[A-Za-z0-9]*.[A-Za-z0-9]*\/[A-Za-z0-9]*|\n' \
        r'twitter.com\/[A-Za-z0-9]*|www.[A-Za-z0-9]*.[A-Za-z0-9]*\/[A-Za-z0-9].*|com|twitter'
cv = CountVectorizer(min_df=5, max_df=0.95)
tf = TfidfTransformer()
n_components = 13
n_top_words = 12
m_lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                  learning_method='online',
                                  learning_offset=50.,
                                  random_state=0)


def import_tweets(filename, regex_string):
    """ Reading JSON file into a list and cleaning the tweets with regex """
    tweets = pd.read_json(f'{filename}.json')['text'].tolist()
    tweets = list(re.sub(regex_string, '', k, flags=re.MULTILINE) for k in tweets)
    return tweets


def spacify_my_text(tweet):
    """ using spacy to retrieve only words that give the sentence a meaning """
    spacyfied = []
    for sentence in tweet:
        parsed_sentence = nlp(sentence.lower())
        treated_sentence = ''
        for token in parsed_sentence:
            if not token.is_stop:
                treated_sentence += str(token.lemma_) + ' '
        spacyfied.append(treated_sentence.strip())
    return spacyfied


def fit_model(cleaned_text):
    """ Transforming list into a binary matrix and training the model """
    fit_vec = cv.fit_transform(cleaned_text)
    fit_tf = tf.fit_transform(fit_vec)
    return m_lda.fit(fit_tf)


def print_top_words2(model, top_words):
    """ Printing main topics from the tweets"""
    topics = []
    feature_names = cv.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-top_words - 1:-1]])
        topics.append(message)
    return topics


def save_textfile(topic_list):
    """ Saving topics into a txt file """
    ls = []
    for i in topic_list:
        strings = i + "\n"
        ls.append(strings)
    file1 = open("extracted_topics.txt", "w")
    file1.writelines(ls)
    file1.close()
    return


tweet_list = import_tweets(file, regex)
spacified_tweets = spacify_my_text(tweet_list)
fitted_model = fit_model(spacified_tweets)

print("\nTopics in LDA model:")
L = print_top_words2(fitted_model, n_top_words)
printed = [print(i) for i in L]
save_textfile(L)
