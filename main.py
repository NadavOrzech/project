import training
import sklearn
# from data_preproccess import create_data_list, create_tfidf
from dataloader import Dataloader
from config import Config

TEST_SIZE = 0.15
RANDOM_STATE = 42


if __name__ == "__main__":
    # words_list, y = create_data_list(input_file)
    config = Config()
    dataloader = Dataloader(config)
    # word1 = create_word2vec(words_list)
    # x_data = create_tfidf(words_list)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataloader.x, dataloader.all_labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    training.bernoulli_model(X_train, X_test, y_train, y_test)
    training.KNN_model(X_train, X_test, y_train, y_test)
    training.SVM_model(X_train, X_test, y_train, y_test)