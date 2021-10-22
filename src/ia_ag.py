import numpy as np
from tqdm.auto import tqdm

import utils

class AlgoritmoGenetico(object):

    def __init__(self, tamano_poblacion, piezas, tamano_tablero, prob_mutar=0.01):
        """
        Init.
        :param tamano_poblacion: tamano de la poblacion.
        :param piezas: lista de piezas representados como poligonos
        :param tamano_tablero: tamanio del tablero de ajedrez
        :param prob_mutar: la probabilidad de ejecutar la mutacion. 1% por default.
        """
        self.tamano_poblacion = tamano_poblacion + 1 if tamano_poblacion % 2 != 0 else tamano_poblacion
        self.piezas = piezas
        self.tamano_tablero = tamano_tablero
        self.prob_mutar = prob_mutar
        self.poblacion = None
        self.mejor = None
        self.generacion = 0


    def seleccion(self):
        """
        Selecciona 2 individuos de la pobliacion de manera aleatoria, pero dando mas peso a los de mejor score,
        de manera que igualmente todos tengan posiblidad de ser elegidos, aunque no con la misma probabilidad.
        :return: dos individuos
        """
        scores = np.array([ind.score for ind in self.poblacion])
        scores = -scores + np.min(scores) + np.max(scores) + 1
        scores = scores / np.sum(scores)
        p1, p2 = np.random.choice(self.poblacion,
                                  size=2,
                                  replace=False,
                                  p=scores)
        return p1, p2

    def mutacion(self):
        """
        Muta a cada individuo usando la problidad de mutar
        """
        for individuo in self.poblacion:
            individuo.mutate(self.prob_mutar)

    def iterar_generacion(self):
        """
        Crea la siguiente generacion, selecciondo y cruzando dos individuos y luego aplicacion mutacion.
        """
        print(f'Generacion: {self.generacion}')
        poblacion_nueva = []
        for _ in tqdm(range(int(self.tamano_poblacion/2)), leave=False):
            p1, p2 = self.seleccion()
            h1, h2 = p1.cruza(p2)
            poblacion_nueva.extend([h1, h2])
        self.poblacion = poblacion_nueva
        self.mutacion()

    def finished(self):
        """
        Devuelve true si ya tenemos la mejor solucion
        """
        scores = np.array([ind.score for ind in self.poblacion], dtype=int)
        if np.any(scores == 0):
            return True

    def mejor_individuo(self):
        """
        Devuelve el mejor individuo
        """
        id = np.argsort([ind.score for ind in self.poblacion])[0]
        return self.poblacion[id]

    def run(self):
        scores_file = open('scores.csv', 'a')
        scores_file.write('generacion,aptitud')
        print(f'Generando problacion inicial..')
        self.poblacion = [Individuo(self.piezas, self.tamano_tablero) for _ in tqdm(range(self.tamano_poblacion), leave=False)]
        self.mutacion()

        self.mejor = self.mejor_individuo()
        self.generacion += 1
        scores_file.write(f"\n{self.generacion}," + f"\n{self.generacion},".join(np.array([ind.score for ind in self.poblacion], dtype=str)))
        print(f'Mejor: {self.mejor.str()}')
        while not self.finished():
            self.iterar_generacion()
            mejor = self.mejor_individuo()
            self.generacion += 1
            scores_file.write(f"\n{self.generacion}," + f"\n{self.generacion},".join(np.array([ind.score for ind in self.poblacion], dtype=str)))
            scores_file.flush()
            if mejor.score < self.mejor.score:
                self.mejor = mejor
            print(f'Mejor: {self.mejor.str()}')
        scores_file.close()

class Individuo(object):
    """
    Un individuo es la representacion del conjunto de todas las piezas ordenadas y acomodadas, en determinada posicion sobre el tablero. 
    """
    def __init__(self, piezas, tamano_tablero, cromosoma=None):
        """
        Initialization function. A random cromosoma is generated if no cromosoma is specified.
        :param piezas: lista de piezas representados como poligonos
        :param tamano_tablero: tamanio del tablero de ajedrez
        :param cromosoma: el cromosoma que representa el orden de las piezas
        """
        self.piezas = piezas
        self.tamano_tablero = tamano_tablero
        self.cromosoma = self.cromosoma_random() if cromosoma is None else np.array(cromosoma)
        self.posiciones, self.score = utils.mejor_posicionamiento(self.cromosoma, piezas, tamano_tablero)

    def cromosoma_random(self):
        return np.random.choice(range(len(self.piezas)), len(self.piezas), replace=False).astype(int)

    def mutate(self, prob_mutar):
        """
        Mutamos intercambiando un gen con un otro gen random,
        solo si se da la condicion de mutar en base a prob_mutar
        """
        cromosoma = self.cromosoma.copy()
        muto = False
        for idx, c in enumerate(cromosoma):
            if np.random.rand() < prob_mutar:
                muto = True
                rand_idx = np.random.choice(range(len(cromosoma)))
                cromosoma[idx], cromosoma[rand_idx] = cromosoma[rand_idx], cromosoma[idx]

        if muto:
            self.cromosoma = cromosoma
            self.posiciones, self.score = utils.mejor_posicionamiento(self.cromosoma, self.piezas, self.tamano_tablero)

    def cruza(self, otro):
        """
        Cruza a si mismo, con otro, utilizando una cruza multipunto, pero generando individuos validos.
        """
        c1, c2 = np.zeros((2, len(self.piezas)), dtype=int)
        start, end = np.sort(np.random.choice(range(len(self.piezas)), 2, replace=False))

        p1 = self.cromosoma
        p2 = otro.cromosoma

        c1[start:end + 1] = p1[start:end + 1]
        mask1 = ~np.in1d(p1, p1[start:end + 1])
        mask2 = ~np.in1d(p2, p1[start:end + 1])
        c1[mask1] = p2[mask2]

        c2[start:end + 1] = p2[start:end + 1]
        mask1 = ~np.in1d(p2, p2[start:end + 1])
        mask2 = ~np.in1d(p1, p2[start:end + 1])
        c2[mask1] = p1[mask2]

        return Individuo(self.piezas, self.tamano_tablero, c1), Individuo(self.piezas, self.tamano_tablero, c2)

    def str(self): 
        """
        Devuelve los detalles de su score, cromosoma y la posicion de las piezas
        """
        res = (f'Score: {self.score}\n'
               f'Cromosoma: [{",".join(np.array(self.cromosoma, dtype=str))}]\n'
               f'Posiciones: \n')
        for idx, c in enumerate(self.cromosoma):
            x = self.posiciones[idx][0][0] if self.posiciones[idx] is not None else '?'
            y = self.posiciones[idx][0][1] if self.posiciones[idx] is not None else '?'
            rot = self.posiciones[idx][1] if self.posiciones[idx] is not None else '?'
            flip = 'Si' if (self.posiciones[idx][2] if self.posiciones[idx] is not None else '?') == -1 else 'No'
            res += f'\tPieza: {c} \t X: {x} Y: {y} \t Rotacion: {rot} \t Invertida: {flip}\n'
        return res
