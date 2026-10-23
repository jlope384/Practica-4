"""visual.py - Dibujo del grafo dirigido (matplotlib)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from red import BODEGA, DESTINOS

C_DOBLE = "#7a8696"
C_ASIM = "#c47f17"
C_UNO = "#c0392b"
C_BODEGA = "#1f6fb2"
C_DEST = "#2e8b57"
C_NODO = "#ffffff"


def posiciones(G):
    """Los identificadores N01..N40 siguen una retícula de 5 filas x 8 columnas
    (N01-N09, N02-N10, ... son tramos 'verticales'). Se usa esa retícula como
    disposición y se desplazan ligeramente las filas impares para que los tramos
    que saltan una fila (p. ej. N11-N27) no se tapen con el nodo intermedio."""
    pos = {}
    for n in G:
        k = int(n[1:]) - 1
        fila, col = divmod(k, 8)
        pos[n] = (col * 2.0 + (0.9 if fila % 2 else 0.0), -fila * 2.2)
    return pos


def _obstruye(pos, u, v, r=0.55):
    """True si el segmento u-v pasa cerca de otro nodo (para curvarlo)."""
    (x1, y1), (x2, y2) = pos[u], pos[v]
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    for n, (x, y) in pos.items():
        if n in (u, v):
            continue
        t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / L2))
        if (x1 + t * dx - x) ** 2 + (y1 + t * dy - y) ** 2 < r * r:
            return True
    return False


def _flecha(ax, p, q, color, lw, rad, estilo="-|>", z=1, alpha=1.0, ms=11):
    a = FancyArrowPatch(p, q, arrowstyle=estilo, mutation_scale=ms, color=color, lw=lw,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=24, shrinkB=24,
                        zorder=z, alpha=alpha)
    ax.add_patch(a)


def _etiqueta(ax, p, q, texto, rad, color, fs=7):
    # punto sobre la curva de Bézier cuadrática que usa arc3; en tramos largos
    # (que saltan una fila) se coloca a 1/3 del origen para no chocar con otros.
    dx, dy = q[0] - p[0], q[1] - p[1]
    cx, cy = (p[0] + q[0]) / 2 + rad * dy, (p[1] + q[1]) / 2 - rad * dx
    t = 0.32 if abs(dy) > 3 else 0.5
    bx = (1 - t) ** 2 * p[0] + 2 * (1 - t) * t * cx + t * t * q[0]
    by = (1 - t) ** 2 * p[1] + 2 * (1 - t) * t * cy + t * t * q[1]
    ax.text(bx, by, texto, fontsize=fs, color=color, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85), zorder=4)


def dibujar_grafo(G, resaltar: dict | None = None, titulo=None, ruta_archivo=None,
                  figsize=(15, 9.5), cerradas=None):
    """resaltar: {nombre: (lista_de_nodos, color)} para pintar rutas encima."""
    pos = posiciones(G)
    fig, ax = plt.subplots(figsize=figsize)
    ab = 0.3 if resaltar else 1.0  # atenuar la red base cuando se resaltan rutas
    hechos = set()
    for u, vs in G.items():
        for v, w in vs.items():
            if (u, v) in hechos:
                continue
            p, q = pos[u], pos[v]
            obst = _obstruye(pos, u, v)
            if u in G.get(v, {}):
                w2 = G[v][u]
                hechos |= {(u, v), (v, u)}
                if w2 == w:
                    r0 = 0.3 if obst else 0.0
                    _flecha(ax, p, q, C_DOBLE, 1.3, r0, estilo="<|-|>", ms=8, alpha=ab)
                    _etiqueta(ax, p, q, f"{w:g}", r0, "#333")
                else:
                    r1 = 0.32 if obst else 0.16
                    _flecha(ax, p, q, C_ASIM, 1.3, r1, ms=10, alpha=ab)
                    _flecha(ax, q, p, C_ASIM, 1.3, r1, ms=10, alpha=ab)
                    _etiqueta(ax, p, q, f"{w:g}", r1, C_ASIM)
                    _etiqueta(ax, q, p, f"{w2:g}", r1, C_ASIM)
            else:
                hechos.add((u, v))
                r2 = 0.3 if obst else 0.0
                _flecha(ax, p, q, C_UNO, 1.6, r2, ms=13, alpha=ab)
                _etiqueta(ax, p, q, f"{w:g}", r2, C_UNO)
    for (u, v) in (cerradas or []):
        p, q = pos[u], pos[v]
        _flecha(ax, p, q, "black", 2.2, 0.0, estilo="-", z=2)
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2, "X", fontsize=14, color="black",
                ha="center", va="center", zorder=6)
    if resaltar:
        for i, (nombre, (ruta, color)) in enumerate(resaltar.items()):
            rad = [0.0, 0.12, -0.12, 0.24][i % 4]
            for a, b in zip(ruta, ruta[1:]):
                extra = 0.3 if _obstruye(pos, a, b) else 0.0
                _flecha(ax, pos[a], pos[b], color, 3.6, rad + extra, z=3, alpha=0.8, ms=16)
            ax.plot([], [], color=color, lw=3.6, label=nombre)
    for n, (x, y) in pos.items():
        if n == BODEGA:
            fc, ec, tc = C_BODEGA, C_BODEGA, "white"
        elif n in DESTINOS:
            fc, ec, tc = C_DEST, C_DEST, "white"
        else:
            fc, ec, tc = C_NODO, "#444", "#222"
        ax.add_patch(plt.Circle((x, y), 0.36, fc=fc, ec=ec, lw=1.4, zorder=5))
        ax.text(x, y, n, fontsize=8, ha="center", va="center", color=tc, weight="bold", zorder=6)
    # leyenda de tipos de tramo
    ax.plot([], [], color=C_DOBLE, lw=1.5, label="Doble sentido, mismo tiempo")
    ax.plot([], [], color=C_ASIM, lw=1.5, label="Doble sentido, tiempos distintos (cada flecha con su tiempo)")
    ax.plot([], [], color=C_UNO, lw=1.8, label="Un solo sentido (→)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=9, frameon=False)
    xs = [x for x, _ in pos.values()]
    ys = [y for _, y in pos.values()]
    ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
    ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
    ax.set_aspect("equal")
    ax.axis("off")
    if titulo:
        ax.set_title(titulo, fontsize=13)
    fig.tight_layout()
    if ruta_archivo:
        fig.savefig(ruta_archivo, dpi=160, bbox_inches="tight")
    return fig
