import training
import sklearn
# from data_preproccess import create_data_list, create_tfidf
from dataloader import Dataloader
from config import Config
from itertools import chain, combinations


TEST_SIZE = 0.15
RANDOM_STATE = 42


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

if __name__ == "__main__":
    
    config = Config()
    dataloader = Dataloader(config,generate_tfidf=False)
    
    x_train, y_train = dataloader.get_train_dataloader()
    
    knn_cv_dict = {
        'model_name': 'KNN',
        'k_list': [1,3,5,7,9,11,13,15,17,19]#,21,25,31,41,51]
    }

    svm_cv_dict = {
        'model_name': 'SVM',
        'c_list': [0.5,1,2,4],
        'kernel_list': ['linear', 'poly', 'rbf', 'sigmoid']
    }
    decision_tree_cv_dict = {
        'model_name': 'DCT',
        'min_samples_split': [1,2,5,10]
    }
    scores = training.cross_validation(x_train,y_train,knn_cv_dict,features_permute=True,powerset_size=7)
    scores = training.cross_validation(x_train,y_train,decision_tree_cv_dict,features_permute=True)
    scores = training.cross_validation(x_train,y_train,svm_cv_dict,features_permute=True)

    aaa=2