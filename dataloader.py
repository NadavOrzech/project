from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import nltk
from config import Config 
import json
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer


class Dataloader():
    def __init__(self, params):
        self.input_path = params.input_path
        self.num_features = len(params.features_map)
        self.features_map = params.features_map
        self.interjections = params.interjections
        self.intensifiers = params.intensifiers

        self.all_words, self.all_labels = self.create_data_list()
        self.features_matrix = np.zeros((len(self.all_words),self.num_features))
        self.extract_raw_features()
        self.extract_initial_process_features()
        self.x = np.concatenate((self.features_matrix,self.tfidf_matrix), axis=1)
        # self.tfidf_matrix = 
        aaa =3

    def create_data_list(self):
        with open(self.input_path, 'r') as file_json:
            data = json.load(file_json)
        
        all_words = []
        all_labels = []
        for key in data:
            headline = data[key]['headline'].lower()
            
            all_words.append(headline)
            all_labels.append(data[key]['is_sarcastic'])

        return all_words, all_labels

    def extract_raw_features(self):
        for i,sentence in enumerate(self.all_words):
            punctuations = sentence.count('?') + sentence.count('!')
            self.features_matrix[i,self.features_map['punctuations']] = punctuations

            quotes = sentence.count('\"') + sentence.count('\'')
            self.features_matrix[i,self.features_map['quotes']] = quotes

            text =  TextBlob(sentence)
            self.features_matrix[i,self.features_map['polarity']] = text.sentiment.polarity
            self.features_matrix[i,self.features_map['subjectivity']] = text.sentiment.subjectivity

    def extract_initial_process_features(self):
        #TODO: consider removeing stop words
        all_words_list = []
        for i, sentence in enumerate(self.all_words):
            processed_article = re.sub('[^a-zA-Z]', ' ', sentence)
            processed_article = re.sub(r'\s+', ' ', processed_article)
            words_list = nltk.word_tokenize(processed_article) 
            intersifier_count, interjection_count = 0,0
            for word in words_list:
                if word in self.intensifiers:
                    intersifier_count+=1
                if word in self.interjections:
                    interjection_count+=1
            self.features_matrix[i,self.features_map['intersifier']] = intersifier_count
            self.features_matrix[i,self.features_map['interjection']] = interjection_count
            self.features_matrix[i,self.features_map['sentence_length']] = len(words_list)
            all_words_list.append(words_list)
        self.tfidf_matrix = create_tfidf(all_words_list)


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


# nltk.download('movie_reviews')
# nltk.download('punkt')
if __name__ == "__main__":
    params = Config()
    dataloader = Dataloader(params)

    


    # text          = "I feel the product is so good" 
    # # text = "mother comes pretty close to using word 'streaming' correctly"
    # sent          = TextBlob(text)
    # # The polarity score is a float within the range [-1.0, 1.0]
    # # where negative value indicates negative text and positive
    # # value indicates that the given text is positive.
    # polarity      = sent.sentiment.polarity
    # # The subjectivity is a float within the range [0.0, 1.0] where
    # # 0.0 is very objective and 1.0 is very subjective.
    # subjectivity  = sent.sentiment.subjectivity

    # sent          = TextBlob(text, analyzer = NaiveBayesAnalyzer())
    # classification= sent.sentiment.classification
    # positive      = sent.sentiment.p_pos
    # negative      = sent.sentiment.p_neg

    # print(polarity,subjectivity,classification,positive,negative)