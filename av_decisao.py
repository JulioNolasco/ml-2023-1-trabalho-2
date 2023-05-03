from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier


def arvore_decisao(X_train, y_train, X_test, y_test):
    # Cria o modelo
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X_train, y_train)

    y_pred_tree = tree.predict(X_test)


    print('\n\nArvore de Decisao')
    print('Matriz Confusao:\n', confusion_matrix(y_test, y_pred_tree))
    print('Acuracy:', accuracy_score(y_test, y_pred_tree))
    print('Precisao:', precision_score(y_test, y_pred_tree))
    print('Revocacao:', recall_score(y_test, y_pred_tree))
