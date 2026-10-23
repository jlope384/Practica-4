"""
simulacion.py - Simulación de eventos discretos de la operación de despacho.

Eventos: LLEGADA (pedido), SALIDA (vehículo sale de N01), ENTREGA (llega al destino
y termina la descarga), REGRESO (vehículo vuelve a N01 y queda libre).
Cola FIFO, cada vehículo atiende un solo pedido por viaje.

Números aleatorios comunes (CRN): los pedidos (instantes, destinos y factores de
tráfico) se generan una sola vez por semilla con `generar_pedidos` y se pasan
idénticos a ambas políticas de ruta.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from red import DESTINOS, BODEGA, rutas_actuales, rutas_recomendadas, ruta_optima, tiempo_ruta

ESCENARIOS = {"Normal": 4.0, "Semana Santa": 6.0, "Navidad": 10.0}
UMBRAL_PUNTUAL = 75.0  # minutos desde la solicitud hasta la entrega


@dataclass
class Pedido:
    id: int
    llegada: float          # minuto en que se recibe la solicitud
    destino: str
    f_ida: float = 1.0      # factor multiplicativo de tráfico (1 = tiempo de la matriz)
    f_reg: float = 1.0
    # resultados
    salida: float = math.nan
    entrega: float = math.nan
    regreso: float = math.nan
    vehiculo: int = -1

    @property
    def espera(self):
        return self.salida - self.llegada

    @property
    def tiempo_sistema(self):
        return self.entrega - self.llegada

    @property
    def puntual(self):
        return self.tiempo_sistema <= UMBRAL_PUNTUAL + 1e-9


def generar_pedidos(tasa_hora: float, horas: float, semilla: int,
                    destinos=DESTINOS, cv_trafico: float = 0.0):
    """Proceso de Poisson homogéneo: tiempos entre llegadas ~ Exponencial(media 60/tasa).
    Destino ~ Uniforme discreta sobre los 4 destinos (misma proporción).
    Factor de tráfico ~ Lognormal de media 1 y coeficiente de variación cv_trafico
    (0 = tiempos deterministas de la matriz). Cada componente usa su propio flujo
    aleatorio derivado de la semilla, para que cambiar uno no altere los demás."""
    ss = np.random.SeedSequence(semilla)
    r_lleg, r_dest, r_traf = (np.random.default_rng(s) for s in ss.spawn(3))
    T = horas * 60.0
    media = 60.0 / tasa_hora
    tiempos, t = [], 0.0
    while True:
        t += r_lleg.exponential(media)
        if t >= T:
            break
        tiempos.append(t)
    n = len(tiempos)
    dest = r_dest.choice(destinos, size=n)
    if cv_trafico > 0:
        s2 = math.log(1 + cv_trafico ** 2)
        f = r_traf.lognormal(-s2 / 2, math.sqrt(s2), size=(n, 2))
    else:
        f = np.ones((n, 2))
    return [Pedido(i + 1, float(tiempos[i]), str(dest[i]), float(f[i, 0]), float(f[i, 1]))
            for i in range(n)]


@dataclass
class Resultado:
    pedidos: list
    n_vehiculos: int
    horas: float
    fin: float                      # minuto en que el último vehículo regresa a N01
    ocupado: float                  # minutos-vehículo ocupados
    cola_t: list = field(default_factory=list)   # (t, largo de cola)
    eventos: list = field(default_factory=list)  # bitácora (t, tipo, pedido, vehículo)

    def metricas(self):
        p = self.pedidos
        n = len(p)
        T = self.horas * 60
        if n == 0:
            return dict(pedidos=0, puntuales=0, pct_puntual=0, espera_prom=0, espera_max=0,
                        sistema_prom=0, sistema_p90=0, utilizacion=0, util_jornada=0,
                        fin_min=T, tiempo_extra=0, extra_entregas=0, cola_max=0)
        esp = np.array([x.espera for x in p])
        sis = np.array([x.tiempo_sistema for x in p])
        punt = int(sum(x.puntual for x in p))
        horizonte = max(T, self.fin)
        # ocupación dentro de la jornada de recepción (0..T)
        ocup_j = sum(max(0.0, min(x.regreso, T) - min(x.salida, T)) for x in p)
        return dict(
            pedidos=n,
            puntuales=punt,
            pct_puntual=100 * punt / n,
            espera_prom=float(esp.mean()),
            espera_max=float(esp.max()),
            sistema_prom=float(sis.mean()),
            sistema_p90=float(np.percentile(sis, 90)),
            utilizacion=100 * self.ocupado / (self.n_vehiculos * horizonte),
            util_jornada=100 * ocup_j / (self.n_vehiculos * T),
            fin_min=self.fin,
            tiempo_extra=max(0.0, self.fin - T),
            extra_entregas=max(0.0, max(x.entrega for x in p) - T),
            cola_max=max((c for _, c in self.cola_t), default=0),
        )


def simular(pedidos, tiempos_ida: dict, tiempos_reg: dict, n_vehiculos: int,
            horas: float, descarga: float = 0.0, bitacora: bool = False) -> Resultado:
    """Simulación de eventos discretos. `pedidos` se copia para no alterar la
    lista compartida entre políticas (CRN)."""
    pedidos = [Pedido(p.id, p.llegada, p.destino, p.f_ida, p.f_reg) for p in pedidos]
    LLEGADA, REGRESO = 0, 1  # prioridad en empates: primero liberar el vehículo
    fel = []  # future event list: (t, prioridad, secuencia, tipo, dato)
    seq = 0

    def programar(t, prioridad, tipo, dato):
        nonlocal seq
        heapq.heappush(fel, (t, prioridad, seq, tipo, dato))
        seq += 1

    for p in pedidos:
        programar(p.llegada, 1, "LLEGADA", p)

    libres = list(range(n_vehiculos))  # vehículos disponibles en N01
    heapq.heapify(libres)
    cola: list[Pedido] = []
    ocupado = 0.0
    fin = 0.0
    cola_t = [(0.0, 0)]
    eventos = []

    def despachar(t):
        nonlocal ocupado
        while libres and cola:
            p = cola.pop(0)
            v = heapq.heappop(libres)
            p.vehiculo = v
            p.salida = t
            t_ida = tiempos_ida[p.destino] * p.f_ida
            t_reg = tiempos_reg[p.destino] * p.f_reg
            p.entrega = t + t_ida + descarga
            p.regreso = p.entrega + t_reg
            ocupado += p.regreso - p.salida
            if bitacora:
                eventos.append((t, "SALIDA", p.id, v))
                eventos.append((p.entrega, "ENTREGA", p.id, v))
            programar(p.regreso, 0, "REGRESO", (p, v))
        cola_t.append((t, len(cola)))

    while fel:
        t, _, _, tipo, dato = heapq.heappop(fel)
        if tipo == "LLEGADA":
            cola.append(dato)
            if bitacora:
                eventos.append((t, "LLEGADA", dato.id, -1))
        else:  # REGRESO
            p, v = dato
            heapq.heappush(libres, v)
            fin = max(fin, t)
            if bitacora:
                eventos.append((t, "REGRESO", p.id, v))
        despachar(t)

    if bitacora:
        eventos.sort(key=lambda e: e[0])
    return Resultado(pedidos, n_vehiculos, horas, fin, ocupado, cola_t, eventos)


# --------------------------------------------------------------------------- #
# Políticas de rutas
# --------------------------------------------------------------------------- #
def reparar_ruta(G, ruta):
    """Si la ruta registrada usa un tramo cerrado, el conductor la sigue hasta el
    último punto alcanzable y desde ahí toma la ruta más corta (desvío)."""
    for i, (u, v) in enumerate(zip(ruta, ruta[1:])):
        if v not in G.get(u, {}):
            desvio, _ = ruta_optima(G, u, ruta[-1])
            if desvio is None:
                return None
            return ruta[:i] + desvio
    return ruta


def politica(G, nombre: str):
    """Devuelve dict destino -> {ida, regreso, t_ida, t_regreso} para la política."""
    if nombre == "Actual":
        base = rutas_actuales(G)
        res = {}
        for d, r in base.items():
            ida = reparar_ruta(G, r["ida"])
            reg = reparar_ruta(G, r["regreso"])
            res[d] = {"ida": ida, "regreso": reg,
                      "t_ida": tiempo_ruta(G, ida) if ida else math.inf,
                      "t_regreso": tiempo_ruta(G, reg) if reg else math.inf,
                      "desvio": ida != r["ida"] or reg != r["regreso"]}
        return res
    res = rutas_recomendadas(G)
    for d in res:
        res[d]["desvio"] = False
    return res


def correr(G, nombre_politica, pedidos, n_vehiculos, horas, descarga=0.0, bitacora=False):
    pol = politica(G, nombre_politica)
    ti = {d: pol[d]["t_ida"] for d in pol}
    tr = {d: pol[d]["t_regreso"] for d in pol}
    if any(math.isinf(x) for x in list(ti.values()) + list(tr.values())):
        raise ValueError("Algún destino quedó inalcanzable con la red actual.")
    return simular(pedidos, ti, tr, n_vehiculos, horas, descarga, bitacora)
