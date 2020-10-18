import sklearn
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def bernoulli_model(X_train, X_test, y_train, y_test):
    bnb = BernoulliNB()
    bnb.fit(X_train, y_train)
    y_pred = bnb.predict(X_test)
    accuracies = accuracy_score(y_test, y_pred)
    print(f"Bernoulli Accuracy: {accuracies}")


def KNN_model(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies = accuracy_score(y_test, y_pred)
    print(f"KNN Accuracy: {accuracies}")


def SVM_model(X_train, X_test, y_train, y_test):
    svm = SVC(kernel="linear")
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracies = accuracy_score(y_test, y_pred)
    print(f"SVM Accuracy: {accuracies}")


def decision_tree_model(X_train, X_test, y_train, y_test):
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, y_train)
    y_pred = decision_tree.predict(X_test)
    accuracies = accuracy_score(y_test, y_pred)
    print(f"decision_tree Accuracy: {accuracies}")


def random_forest_model(X_train, X_test, y_train, y_test):
    rnd_forest = RandomForestClassifier(n_estimators=50)
    rnd_forest.fit(X_train, y_train)
    y_pred = rnd_forest.predict(X_test)
    accuracies = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracies}")
