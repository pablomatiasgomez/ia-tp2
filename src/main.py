import numpy as np

import ia_ag
import utils


if __name__ == '__main__':

    # 13 piezas, representadas por una lista de puntos de cada una, que corresponde a su 
    # posicion x e y en el tablero y un tercer valor que indica el color del mismo dentro del tablero.
    forma_piezas = np.array([
        [(0, 0, 0), (1, 0, 1), (2, 0, 0), (3, 0, 1), (4, 0, 0)],
        [(0, 0, 0), (1, 0, 1), (2, 0, 0), (2, -1, 1), (1, 1, 0)],
        [(0, 0, 0), (1, 0, 1), (2, 0, 0), (0, 1, 1), (2, -1, 1)],
        [(0, 0, 0), (1, 0, 1), (2, 0, 0), (-1, 0, 1), (0, -1, 1)],
        [(0, 0, 0), (-1, 0, 1), (1, 0, 1), (0, 1, 1), (0, -1, 1)],
        [(0, 0, 1), (1, 0, 0), (2, 0, 1), (0, 1, 0), (0, -1, 0)],
        [(0, 0, 0), (1, 0, 1), (2, 0, 0), (3, 0, 1), (0, 1, 1)],
        [(0, 0, 0), (0, 1, 1), (2, 0, 0), (2, 1, 1), (1, 1, 0)],
        [(0, 0, 1), (1, 0, 0), (2, 0, 1), (0, 1, 0), (0, 2, 1)],
        [(0, 0, 1), (-1, 0, 0), (0, 1, 0), (1, 1, 1), (2, 1, 0)],
        [(0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 0), (-1, 1, 0)],
        [(0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1, 1)],
        [(0, 0, 0), (1, 1, 0), (2, 2, 0), (0, 1, 1), (1, 2, 1)]
    ])
    tamano_poblacion = 50
    tamano_tablero = 8
    piezas = utils.crear_poligonos(forma_piezas)

    runner = ia_ag.AlgoritmoGenetico(tamano_poblacion, piezas, tamano_tablero, prob_mutar=0.01)
    runner.run()
