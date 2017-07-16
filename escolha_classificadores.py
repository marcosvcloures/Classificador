
from classifier import *
import warnings

warnings.filterwarnings("ignore")

if(__name__ == '__main__'):
    x, y, height, width, pca = obterDados(dirname(abspath(__file__)) + "\\animais", MEDCINZA)

    for i in estimadores.split(' '):
        print(np.mean(model_selection.cross_val_score(eval(i), x, y, n_jobs = 8)))