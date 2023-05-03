"""
Desenvolvedor:	julio cezar nolasco
E-mail:			juliocenolasco@gmail.com

Código do trabalho 1 com adaptacoes para atender as necessidades do trabalho 2

"""

import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from NB import naive_bayes
from av_decisao import arvore_decisao
from knn import knn

dados = pd.read_csv('dataset.csv')

def drop_column_null(data):
    with open("dataset.csv", "r") as arquivo:
        arquivo_csv = csv.reader(arquivo, delimiter=",")                    #separa as colunas por ","
        for i, coluna in enumerate(arquivo_csv):                            # i == linha (indice)
            if i == 0:
                dados = pd.DataFrame(data)                                  # facilita a vida para trbalhar os dados a diante
                for x in range(len(coluna)):
                    if dados['{}'.format(coluna[x])].isnull().all():        # verifica se a coluna não tem dado
                        dados = dados.drop(columns=coluna[x])               # se sim, apaga
                        # print(coluna[x])
                        # print(dados)
    return dados


def prepara_dados(data):
    with open("dataset.csv", "r") as arquivo:
        arquivo_csv = csv.reader(arquivo, delimiter=",")
        aux = pd.DataFrame(data)
        y = aux['SARS-Cov-2 exam result']
        X = aux.drop(['Patient ID', 'SARS-Cov-2 exam result'], axis=1)
        dados = data.apply(pd.to_numeric, errors='coerce').fillna(value=pd.np.nan)                #pega todos os valores q não são números e atribue um valor nulo
        for i, coluna in enumerate(arquivo_csv):
            if i == 0:
                dados = pd.DataFrame(dados)
                for x in range(len(coluna)):
                    media_coluna = dados['{}'.format(coluna[x])].dropna().mean()                        #pega a média dos valores presentes na coluna
                    dados['{}'.format(coluna[x])] = dados['{}'.format(coluna[x])].fillna(media_coluna)  #atribui a media nos valores que estavam sem valor
    dados = drop_column_null(dados)
    return X, y, dados


X, y, dados = prepara_dados(dados)

##
# Treino e teste
##
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


##
# chamada de cada funcao para fins de comparacao
##
knn(X_train, y_train, X_test, y_test)

naive_bayes(X_train, y_train, X_test, y_test)

arvore_decisao(X_train, y_train, X_test, y_test)

