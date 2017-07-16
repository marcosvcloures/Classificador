
from classifier import *
import warnings

warnings.filterwarnings("ignore")

def desenhos():
    print("Desenhos")

    x, y, height, width, pca = obterDados(dirname(abspath(__file__)) + "\\simpsons", BASICO)
    tx, ty = obterImagens(dirname(abspath(__file__)) + "\\validacaoSimpsons", BASICO, height, width, pca)

    svc_l.fit(x, y)
    rfc_g10.fit(x, y)
    knn_d5.fit(x, y)

    print("SVC LIN")
    
    acertos = 0
    for idx, val in enumerate(tx):
        print(svc_l.predict(val), ty[idx])
        
        if(svc_l.predict(val) == ty[idx]):
            acertos += 1

    print(acertos / len(tx), "\n----")

    print("RFC G10")

    acertos = 0
    for idx, val in enumerate(tx):
        print(rfc_g10.predict(val), ty[idx])

        if(rfc_g10.predict(val) == ty[idx]):
            acertos += 1

    print(acertos / len(tx), "\n----")
    
    print("KNN D5")

    acertos = 0
    for idx, val in enumerate(tx):
        print(knn_d5.predict(val), ty[idx])

        if(knn_d5.predict(val) == ty[idx]):
            acertos += 1

    print(acertos / len(tx), "\n----")

def animais():
    print("Animais")

    x, y, height, width, pca = obterDados(dirname(abspath(__file__)) + "\\animais", BASICO)
    tx, ty = obterImagens(dirname(abspath(__file__)) + "\\validacaoAnimal", BASICO, height, width, pca)

    rfc_e20.fit(x, y)
    knn_u5.fit(x, y)

    print("RFC E20")
    
    acertos = 0
    for idx, val in enumerate(tx):
        print(rfc_e20.predict(val), ty[idx])
        
        if(rfc_e20.predict(val) == ty[idx]):
            acertos += 1

    print(acertos / len(tx), "\n----")

    print("KNN U5")

    acertos = 0
    for idx, val in enumerate(tx):
        print(knn_u5.predict(val), ty[idx])

        if(knn_u5.predict(val) == ty[idx]):
            acertos += 1

    print(acertos / len(tx), "\n----")
    
    x, y, height, width, pca = obterDados(dirname(abspath(__file__)) + "\\animais", MEDCINZA)
    tx, ty = obterImagens(dirname(abspath(__file__)) + "\\validacaoAnimal", MEDCINZA, height, width, pca)

    rfc_g20.fit(x, y)

    print("RFC G20")

    acertos = 0
    for idx, val in enumerate(tx):
        print(rfc_g20.predict(val), ty[idx])

        if(rfc_g20.predict(val) == ty[idx]):
            acertos += 1

    print(acertos / len(tx), "\n----")

desenhos()
animais()