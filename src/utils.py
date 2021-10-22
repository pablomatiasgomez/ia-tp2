import itertools
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point, MultiPoint
from shapely.affinity import translate, rotate, scale
from functools import partial
from multiprocessing import Pool

import warnings
warnings.filterwarnings('ignore')


def crear_poligonos(piezas):
    """
    Crea los poligonos de cada pieza, con todas sus posibles orientaciones
    Las piezas son una lista de tuplas (x, y, m) que representan las coordenanas 
    :param piezas: lista de piezas
    :return: np.array de poligonos
    """
    polygons = []
    for idx, pieza in enumerate(piezas):
        pts = np.array(pieza)[:, :-1]
        poly = MultiPoint(pts+0.5).buffer(0.5, cap_style=3)
        poly.checkers = np.array(pieza)
        poly.orientations = orientaciones_de_pieza(poly)
        poly.id = idx
        polygons.append(poly)
    return np.array(polygons)


def orientaciones_de_pieza(poly):
    """
    Devuelve todas las posibles posiciones de una pieza (poligonos), incluyendo darla vuelta.
    :param poly: poligono que representa la pieza
    :return: array of unique settings of (rotation, flip)
    """
    cx, cy = poly.centroid.x, poly.centroid.y
    pts = np.array(poly.checkers)[:, :-1]
    checkers = np.array(poly.checkers)[:, -1]
    pts = pts[checkers == 1]
    poly = poly.difference(MultiPoint(pts + 0.5).buffer(0.2))

    rots = [0, 90, 180, 270]
    flips = [1, -1]
    iterables = [rots, flips]
    settings = np.array(list(itertools.product(*iterables)))
    unique_polys = []
    unique_settings = []
    for rot, flip in settings:
        p = poly
        p = rotate(p, rot, origin=[cx, cy])
        p = scale(p, flip, origin=[cx, cy, 0])
        if np.any([p.difference(u).area < 1e-5 for u in unique_polys]):
            continue
        unique_polys.append(p)
        unique_settings.append([rot, flip])
    return unique_settings


def mejor_posicionamiento(cromosoma, poligonos, tamano_tablero):
    """
    Para cada pieza del cromosoma, la ubica en su posicion optima hasta obtener el mejor orden
    Devuelve ademas el score
    """
    profile = Polygon([[0, 0], [tamano_tablero, 0], [tamano_tablero, tamano_tablero], [0, tamano_tablero]])
    placements = np.full_like(cromosoma, None, dtype=object)
    origin_checker = None
    for idx, c in enumerate(cromosoma):
        p = poligonos[c]
        try:
            opt, profile = optimal_placement(profile, p, tamano_tablero, origin_checker)
            placements[idx] = opt
            if origin_checker is None:
                sum_vec = np.sum(opt[0])
                if sum_vec % 2 == p.checkers[0, -1]:
                    origin_checker = 0
                else:
                    origin_checker = 1
        except Exception:
            idx -= 1
            break
    empty_area = profile.area
    n_unused_pieces = len(cromosoma) - idx - 1
    outline = calculate_outline(profile)
    fitness_score = empty_area + n_unused_pieces + outline
    return placements, fitness_score


def optimal_placement(profile, poly, tamano_tablero, origin_checker=None):
    point_grid = np.array([Point([x + 0.5, y + 0.5]) for x in range(tamano_tablero) for y in range(tamano_tablero)],
                          dtype=object)
    position_grid = np.array([[x, y] for x in range(tamano_tablero) for y in range(tamano_tablero)], dtype=int)
    point_mask = np.array([pt.within(profile) for pt in point_grid], dtype=bool)

    if origin_checker is not None:
        if origin_checker == poly.checkers[0, -1]:
            point_mask &= np.sum(position_grid, axis=-1) % 2 == 0
        else:
            point_mask &= np.sum(position_grid, axis=-1) % 2 == 1

    rflips = poly.orientations
    pos = position_grid[point_mask]

    iterables = [pos, rflips]
    settings = list(itertools.product(*iterables))
    settings = [(s[0], *s[1]) for s in settings]

    func = partial(get_new_outline, profile, poly)

    pool = Pool(processes=None)
    outlines = np.array(pool.map(func, settings))
    pool.close()

    outlines = outlines.astype(float)
    opt_setting = settings[np.argsort(outlines)[0]]
    opt_profile = get_new_profile(profile, poly, opt_setting)

    if opt_profile is not None:
        return opt_setting, opt_profile


def get_new_outline(profile, poly, setting):
    return calculate_outline(get_new_profile(profile, poly, setting))


def calculate_outline(profile):
    if profile is None:
        return np.nan

    outline = profile.exterior.length
    for interior in profile.interiors:
        outline += interior.length
    return outline


def get_new_profile(profile, poly, setting):
    poly = transform(poly, *setting)
    if not poly.within(profile):
        return None

    new_profile = profile.difference(poly)
    if type(new_profile) == MultiPolygon:
        return None

    return new_profile


def transform(poly, vec, r, f):
    poly = rotate(poly, r, origin=np.array([0.5, 0.5]))
    poly = scale(poly, f, origin=np.array([0.5, 0.5, 0.0]))
    poly = translate(poly, *vec)
    return poly

