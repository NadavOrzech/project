import training
import sklearn
from data_preproccess import create_data_list, create_tfidf


input_file = ".\\sarcasm_dataset_small.json"
TEST_SIZE = 0.15
RANDOM_STATE = 42


if __name__ == "__main__":
    words_list, y = create_data_list(input_file)

    # word1 = create_word2vec(words_list)
    x_data = create_tfidf(words_list)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x_data, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    # training.bernoulli_model(X_train, X_test, y_train, y_test)
    # training.KNN_model(X_train, X_test, y_train, y_test)
    training.SVM_model(X_train, X_test, y_train, y_test)