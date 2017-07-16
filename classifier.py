import numpy as np
from sklearn import decomposition, neural_network, svm, ensemble, neighbors, model_selection
import cv2
import random
from os import listdir
from os.path import dirname, abspath, isfile, join

BASICO = 0
CINZA = 2
LIMIAR = 6
MEDIANA = 1
MEDCINZA = 3
MEDLIMIAR = 7

def apicarFiltros(img, filtros):
    if filtros & 1: # mediana
        img = cv2.medianBlur(img, 5)

    if filtros & 2: # tons de cinza
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if filtros & 4: # limiar global adaptativa
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    if filtros & 2: # voltando do tons de cinza
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img

def obterImagens(caminho, filtros, height, width, pca):
    arquivos = [cv2.imread(join(caminho, f), 1) for f in listdir(caminho) if isfile(join(caminho, f))]
    nomes = [f.split('-')[1].split('.')[0] for f in listdir(caminho) if isfile(join(caminho, f))]

    arquivos = [apicarFiltros(f, filtros) for f in arquivos]

    arquivos = [cv2.resize(f, (width, height), interpolation = cv2.INTER_CUBIC) for f in arquivos]

    arquivos = [np.reshape(f, width * height * 3) for f in arquivos]

    base = pca.transform(arquivos)

    return base, nomes


def obterDados(caminho, filtros):
    arquivos = [cv2.imread(join(caminho, f), 1) for f in listdir(caminho) if isfile(join(caminho, f))]
    nomes = [f.split('-')[1].split('.')[0] for f in listdir(caminho) if isfile(join(caminho, f))]

    min_height = min_width = 1 << 30

    for i in arquivos:
        min_height = min(i.shape[0], min_height)
        min_width = min(i.shape[1], min_width)

    arquivos = [apicarFiltros(f, filtros) for f in arquivos]

    arquivos = [cv2.resize(f, (min_width, min_height), interpolation = cv2.INTER_CUBIC) for f in arquivos]

    ruido = np.zeros((min_height, min_width, 3), np.uint8)

    for i in range(0, len(arquivos)): 
        if random.random() <= 0.3: #ruido gaussiano
            cv2.randn(ruido, (0, 0, 0), (20, 20, 20))
            arquivos.append(arquivos[i] + ruido)
            nomes.append(nomes[i])
        if random.random() <= 0.3: #vertical
            arquivos.append(cv2.flip(arquivos[i], 1))
            nomes.append(nomes[i])
        if random.random() <= 0.3: #horizontal
            arquivos.append(cv2.flip(arquivos[i], 0))
            nomes.append(nomes[i])
    
    arquivos = [np.reshape(f, min_width * min_height * 3) for f in arquivos]

    pca = decomposition.PCA(25)

    base = pca.fit_transform(arquivos)

    return base, nomes, min_height, min_width, pca

mlpc_100 = neural_network.MLPClassifier(random_state = 3, hidden_layer_sizes = (100,))
mlpc_50 = neural_network.MLPClassifier(random_state = 3, hidden_layer_sizes = (50,))
mlpc_10 = neural_network.MLPClassifier(random_state = 3, hidden_layer_sizes = (10,))

svc_l = svm.SVC(random_state = 3, max_iter = 500, kernel = 'linear')
svc_r = svm.SVC(random_state = 3, max_iter = 500, kernel = 'rbf')
svc_3 = svm.SVC(random_state = 3, max_iter = 500, kernel = 'poly', degree = 3)
svc_5 = svm.SVC(random_state = 3, max_iter = 500, kernel = 'poly', degree = 5)

rfc_e5 = ensemble.RandomForestClassifier(random_state = 3, n_jobs = 4, n_estimators = 5, criterion = 'entropy')
rfc_e10 = ensemble.RandomForestClassifier(random_state = 3, n_jobs = 4, n_estimators = 10, criterion = 'entropy')
rfc_e20 = ensemble.RandomForestClassifier(random_state = 3, n_jobs = 4, n_estimators = 20, criterion = 'entropy')

rfc_g5 = ensemble.RandomForestClassifier(random_state = 3, n_jobs = 4, n_estimators = 5, criterion = 'gini')
rfc_g10 = ensemble.RandomForestClassifier(random_state = 3, n_jobs = 4, n_estimators = 10, criterion = 'gini')
rfc_g20 = ensemble.RandomForestClassifier(random_state = 3, n_jobs = 4, n_estimators = 20, criterion = 'gini')

knn_u5 = neighbors.KNeighborsClassifier(n_jobs = 4, n_neighbors = 5, weights = 'uniform')
knn_u10 = neighbors.KNeighborsClassifier(n_jobs = 4, n_neighbors = 10, weights = 'uniform')
knn_u20 = neighbors.KNeighborsClassifier(n_jobs = 4, n_neighbors = 20, weights = 'uniform')

knn_d5 = neighbors.KNeighborsClassifier(n_jobs = 4, n_neighbors = 5, weights = 'distance')
knn_d10 = neighbors.KNeighborsClassifier(n_jobs = 4, n_neighbors = 10, weights = 'distance')
knn_d20 = neighbors.KNeighborsClassifier(n_jobs = 4, n_neighbors = 20, weights = 'distance')

estimadores = "mlpc_100 mlpc_50 mlpc_10 svc_l svc_r svc_3 svc_5 rfc_e5 rfc_e10 rfc_e20 rfc_g5 rfc_g10 rfc_g20 knn_u5 knn_u10 knn_u20 knn_d5 knn_d10 knn_d20"