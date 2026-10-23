"""
red.py - Construcción del grafo dirigido y algoritmos de rutas.
Logística del Valle - CC2017 Modelación y Simulación, Práctica 4.
"""
from __future__ import annotations

import csv
import heapq
import math
from collections import deque
from pathlib import Path

BODEGA = "N01"
DESTINOS = ["N08", "N19", "N34", "N40"]

# Rutas registradas en las hojas de despacho (ida). El regreso es la misma secuencia invertida.
RUTAS_ACTUALES = {
    "N08": ["N01", "N02", "N03", "N04", "N05", "N06", "N07", "N08"],
    "N19": ["N01", "N02", "N03", "N11", "N19"],
    "N34": ["N01", "N09", "N17", "N25", "N33", "N34"],
    "N40": ["N01", "N02", "N03", "N04", "N05", "N06", "N07", "N08",
            "N16", "N24", "N32", "N40"],
}


# --------------------------------------------------------------------------- #
# Carga del grafo
# --------------------------------------------------------------------------- #
def cargar_grafo(ruta_csv: str | Path) -> dict[str, dict[str, float]]:
    """Lee la matriz de adyacencia (fila = origen, columna = destino).
    Celda vacía o guion = sin conexión directa; 0 en la diagonal = mismo punto.
    Devuelve una lista de adyacencia dirigida: G[u][v] = minutos de u -> v."""
    with open(ruta_csv, newline="", encoding="utf-8-sig") as f:
        filas = list(csv.reader(f))
    columnas = [c.strip() for c in filas[0][1:]]
    G: dict[str, dict[str, float]] = {c: {} for c in columnas}
    for fila in filas[1:]:
        if not fila or not fila[0].strip():
            continue
        origen = fila[0].strip()
        G.setdefault(origen, {})
        for destino, celda in zip(columnas, fila[1:]):
            celda = celda.strip()
            if celda in ("", "—", "-", "–") or destino == origen:
                continue
            G[origen][destino] = float(celda)
    return G


def copiar(G):
    return {u: dict(vs) for u, vs in G.items()}


def aristas(G):
    return [(u, v, w) for u, vs in G.items() for v, w in vs.items()]


def clasificar_aristas(G):
    """Separa los tramos en: doble sentido con igual tiempo, doble sentido con
    tiempos distintos y de un solo sentido."""
    simetricas, asimetricas, un_sentido = [], [], []
    vistos = set()
    for u, v, w in aristas(G):
        if (v, u) in vistos:
            continue
        vistos.add((u, v))
        if u in G.get(v, {}):
            (simetricas if G[v][u] == w else asimetricas).append((u, v, w, G[v][u]))
        else:
            un_sentido.append((u, v, w))
    return simetricas, asimetricas, un_sentido


# --------------------------------------------------------------------------- #
# Validación de rutas
# --------------------------------------------------------------------------- #
def evaluar_ruta(G, ruta):
    """Devuelve (valida, tiempo_total, detalle). detalle = lista (u, v, w|None)."""
    detalle, total, valida = [], 0.0, True
    for u, v in zip(ruta, ruta[1:]):
        w = G.get(u, {}).get(v)
        detalle.append((u, v, w))
        if w is None:
            valida = False
        else:
            total += w
    return valida, (total if valida else math.inf), detalle


def tiempo_ruta(G, ruta):
    return evaluar_ruta(G, ruta)[1]


# --------------------------------------------------------------------------- #
# Algoritmos de ruta más corta
# --------------------------------------------------------------------------- #
def dijkstra(G, origen, destino=None):
    """Dijkstra con cola de prioridad (heap binario). O((V+E) log V).
    Si se da destino se detiene al extraerlo. Devuelve (dist, previo, expandidos)."""
    dist = {origen: 0.0}
    previo = {origen: None}
    visitados = set()
    heap = [(0.0, origen)]
    expandidos = 0
    while heap:
        d, u = heapq.heappop(heap)
        if u in visitados:
            continue
        visitados.add(u)
        expandidos += 1
        if u == destino:
            break
        for v, w in G.get(u, {}).items():
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                previo[v] = u
                heapq.heappush(heap, (nd, v))
    return dist, previo, expandidos


def reconstruir(previo, destino):
    if destino not in previo:
        return None
    camino, nodo = [], destino
    while nodo is not None:
        camino.append(nodo)
        nodo = previo[nodo]
    return camino[::-1]


def ruta_optima(G, origen, destino):
    dist, previo, _ = dijkstra(G, origen, destino)
    camino = reconstruir(previo, destino)
    return camino, dist.get(destino, math.inf)


def heuristica_saltos(G, destino):
    """Heurística admisible para A*: h(n) = (mínimo de saltos de n al destino) x
    (peso mínimo de cualquier tramo). No hay coordenadas en los datos, por lo que
    se usa la estructura del grafo. Nunca sobreestima -> A* sigue siendo óptimo."""
    w_min = min(w for _, _, w in aristas(G))
    inverso = {}
    for u, v, _ in aristas(G):
        inverso.setdefault(v, []).append(u)
    saltos = {destino: 0}
    q = deque([destino])
    while q:
        x = q.popleft()
        for p in inverso.get(x, []):
            if p not in saltos:
                saltos[p] = saltos[x] + 1
                q.append(p)
    return {n: saltos.get(n, math.inf) * w_min for n in G}


def a_estrella(G, origen, destino, h=None):
    h = h or heuristica_saltos(G, destino)
    g = {origen: 0.0}
    previo = {origen: None}
    cerrados = set()
    heap = [(h.get(origen, 0), 0.0, origen)]
    expandidos = 0
    while heap:
        f, d, u = heapq.heappop(heap)
        if u in cerrados:
            continue
        cerrados.add(u)
        expandidos += 1
        if u == destino:
            break
        for v, w in G.get(u, {}).items():
            nd = d + w
            if nd < g.get(v, math.inf):
                g[v] = nd
                previo[v] = u
                heapq.heappush(heap, (nd + h.get(v, 0), nd, v))
    return reconstruir(previo, destino), g.get(destino, math.inf), expandidos


def bellman_ford(G, origen):
    nodos = list(G)
    dist = {n: math.inf for n in nodos}
    previo = {n: None for n in nodos}
    dist[origen] = 0.0
    relajaciones = 0
    for _ in range(len(nodos) - 1):
        cambio = False
        for u, v, w in aristas(G):
            relajaciones += 1
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                previo[v] = u
                cambio = True
        if not cambio:
            break
    previo = {k: v for k, v in previo.items() if v is not None or k == origen}
    return dist, previo, relajaciones


def floyd_warshall(G):
    nodos = list(G)
    D = {i: {j: (0.0 if i == j else G[i].get(j, math.inf)) for j in nodos} for i in nodos}
    for k in nodos:
        Dk = D[k]
        for i in nodos:
            dik = D[i][k]
            if dik == math.inf:
                continue
            Di = D[i]
            for j in nodos:
                if dik + Dk[j] < Di[j]:
                    Di[j] = dik + Dk[j]
    return D


def k_rutas(G, origen, destino, k=3):
    """Algoritmo de Yen: k rutas simples más cortas (usa Dijkstra como subrutina)."""
    camino, costo = ruta_optima(G, origen, destino)
    if camino is None:
        return []
    A = [(costo, camino)]
    B = []
    for _ in range(1, k):
        ultimo = A[-1][1]
        for i in range(len(ultimo) - 1):
            espuela = ultimo[i]
            raiz = ultimo[: i + 1]
            H = copiar(G)
            for _, p in A:
                if p[: i + 1] == raiz and len(p) > i + 1:
                    H[p[i]].pop(p[i + 1], None)
            for n in raiz[:-1]:
                H[n] = {}
                for vs in H.values():
                    vs.pop(n, None)
            c_esp, costo_esp = ruta_optima(H, espuela, destino)
            if c_esp is None:
                continue
            total = raiz[:-1] + c_esp
            t = tiempo_ruta(G, total)
            if all(total != p for _, p in A) and all(total != p for _, p in B):
                heapq.heappush(B, (t, total))
        if not B:
            break
        A.append(heapq.heappop(B))
    return A


def rutas_recomendadas(G, origen=BODEGA, destinos=DESTINOS):
    """Ruta de ida (origen->d) y de regreso (d->origen) calculadas por Dijkstra."""
    res = {}
    for d in destinos:
        ida, t_ida = ruta_optima(G, origen, d)
        reg, t_reg = ruta_optima(G, d, origen)
        res[d] = {"ida": ida, "t_ida": t_ida, "regreso": reg, "t_regreso": t_reg}
    return res


def rutas_actuales(G, destinos=DESTINOS):
    res = {}
    for d in destinos:
        ida = RUTAS_ACTUALES[d]
        reg = ida[::-1]
        res[d] = {"ida": ida, "t_ida": tiempo_ruta(G, ida),
                  "regreso": reg, "t_regreso": tiempo_ruta(G, reg)}
    return res
