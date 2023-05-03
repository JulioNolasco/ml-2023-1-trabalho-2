from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB


def naive_bayes(X_train, y_train, X_test, y_test):
    # Criando o modelo Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)

    y_pred_nb = nb.predict(X_test)

    print('\n\nNaive Bayes')
    print('Matriz de Confusão:\n', confusion_matrix(y_test, y_pred_nb))
    print('Acurácia:', accuracy_score(y_test, y_pred_nb))
    print('Precisão:', precision_score(y_test, y_pred_nb))
    print('Revocação:', recall_score(y_test, y_pred_nb))