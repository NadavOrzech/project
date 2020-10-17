import json
import os
import re
import nltk
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import BernoulliNB
# nltk.download()

input_file = "C:\\Users\\Dell\\Desktop\\Technion\\Project\\news-headlines-dataset-for-sarcasm-detection\\sarcasm_dataset.json"

def create_data_list(file_path):
    with open(file_path,'r') as file_json:
        data = json.load(file_json)
    
    all_words = []
    all_labels = []
    for key in data:
        processed_article = data[key]['headline'].lower()
        processed_article = re.sub('[^a-zA-Z]', ' ', processed_article)
        processed_article = re.sub(r'\s+', ' ', processed_article)

        words_list = nltk.word_tokenize(processed_article) 
        all_words.append(words_list)
        all_labels.append(data[key]['is_sarcastic'])

    return all_words,all_labels 

def create_word2vec(all_words_list):
    word2vec = Word2Vec(all_words_list, min_count=2, sg=1)
    # vocabulary = word2vec.wv.vocab


    return word2vec

def create_tfidf(all_words_list):
    vectorizer = TfidfVectorizer()


    sentences_list = []
    for sentence in all_words_list:
        ps = PorterStemmer()
        sentence = [ps.stem(word) for word in sentence]
        sentences_list.append(' '.join(sentence))

        # if ' '.join(sentece1) != ' '.join(sentece):
        #     print("With stem: "+ ' '.join(sentece1))
        #     print("No stem:   "+' '.join(sentece))

    tfidf = vectorizer.fit_transform(sentences_list).toarray()

    return tfidf


if __name__ == "__main__":
    words_list,y = create_data_list(input_file)
    
    word1 = create_word2vec(words_list)
    word2 = create_tfidf(words_list)

    bnb=BernoulliNB()
    bnb.fit(word2,y)
    # labels_pred = bnb.predict(feature_test)

    # from sklearn.model_selection import cross_val_score
    # accuracies = cross_val_score(estimator = bnb, X = features, y = labels, cv = 10)
    # print ("mean accuracy is",accuracies.mean())
    # print (accuracies.std())
    
    
    
    
    aaa=2