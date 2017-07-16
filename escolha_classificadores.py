
from classifier import *
import warnings

warnings.filterwarnings("ignore")

if(__name__ == '__main__'):
    filtros = ["BASICO", "CINZA", "LIMIAR", "MEDIANA", "MEDCINZA", "MEDLIMIAR"]

    print("ANIMAIS")

    for filtro in filtros:
        print(filtro)

        x, y, height, width, pca = obterDados(dirname(abspath(__file__)) + "\\animais", eval(filtro))

        for estimador in estimadores.split(' '):
            print(estimador, ": ", np.mean(model_selection.cross_val_score(eval(estimador), x, y, n_jobs = 8)))

    print("SIMPSONS")

    for filtro in filtros:
        print(filtro)

        x, y, height, width, pca = obterDados(dirname(abspath(__file__)) + "\\simpsons", eval(filtro))

        for estimador in estimadores.split(' '):
            print(estimador, ": ", np.mean(model_selection.cross_val_score(eval(estimador), x, y, n_jobs = 8)))