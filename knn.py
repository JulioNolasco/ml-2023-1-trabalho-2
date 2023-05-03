from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

def knn(X_train, y_train, X_test, y_test):
    # Criando o modelo KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    y_pred_knn = knn.predict(X_test)

    print('\n\nKNN')
    print('Matriz de Confusão:\n', confusion_matrix(y_test, y_pred_knn))
    print('Acurácia:', accuracy_score(y_test, y_pred_knn))
    print('Precisão:', precision_score(y_test, y_pred_knn))
    print('Revocação:', recall_score(y_test, y_pred_knn))
