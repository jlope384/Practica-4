"""
experimentos.py - Genera todas las tablas (CSV) y figuras (PNG) del informe.
Uso:  python experimentos.py      (resultados en ./resultados y ./figuras)
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from red import (BODEGA, DESTINOS, RUTAS_ACTUALES, a_estrella, bellman_ford, cargar_grafo,
                 copiar, dijkstra, floyd_warshall, k_rutas, reconstruir, rutas_actuales,
                 rutas_recomendadas, tiempo_ruta, clasificar_aristas, evaluar_ruta)
from simulacion import ESCENARIOS, correr, generar_pedidos, politica
from visual import dibujar_grafo

AQUI = Path(__file__).parent
RES = AQUI / "resultados"
FIG = AQUI / "figuras"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

SEMILLAS = list(range(1, 11))   # 10 repeticiones por escenario (mínimo pedido: 5)
VEHICULOS = 8
HORAS = 8
COL_ACT, COL_DIJ = "#c0392b", "#1f6fb2"

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def fmt(r):
    return " → ".join(r) if r else "-"


# --------------------------------------------------------------------------- #
def validar_actuales(G):
    filas = []
    for d, ruta in RUTAS_ACTUALES.items():
        for sentido, r in (("Ida", ruta), ("Regreso", ruta[::-1])):
            ok, t, det = evaluar_ruta(G, r)
            filas.append({"destino": d, "sentido": sentido, "ruta": fmt(r), "tramos": len(det),
                          "valida": "Sí" if ok else "No",
                          "tramos_inexistentes": ", ".join(f"{u}→{v}" for u, v, w in det if w is None) or "-",
                          "detalle": " + ".join(f"{w:g}" for *_, w in det if w is not None),
                          "tiempo_min": t})
    df = pd.DataFrame(filas)
    df.to_csv(RES / "validacion_rutas_actuales.csv", index=False)
    return df


def tabla_alternativas(G, k=3):
    """Ruta actual vs. k rutas más cortas (Yen; la 1.ª es la de Dijkstra), por sentido."""
    filas = []
    for d in DESTINOS:
        for sentido, (o, t) in (("Ida", (BODEGA, d)), ("Regreso", (d, BODEGA))):
            actual = RUTAS_ACTUALES[d] if sentido == "Ida" else RUTAS_ACTUALES[d][::-1]
            t_act = tiempo_ruta(G, actual)
            filas.append({"destino": d, "sentido": sentido, "opcion": "Actual",
                          "ruta": fmt(actual), "tiempo_min": t_act, "dif_vs_actual": 0.0})
            for i, (c, r) in enumerate(k_rutas(G, o, t, k + 1)):
                if r == actual:
                    etiqueta = f"Alternativa {i+1} (= actual)"
                else:
                    etiqueta = "Dijkstra (óptima)" if i == 0 else f"Alternativa {i+1} (Yen)"
                filas.append({"destino": d, "sentido": sentido, "opcion": etiqueta,
                              "ruta": fmt(r), "tiempo_min": c, "dif_vs_actual": c - t_act})
    df = pd.DataFrame(filas)
    df.to_csv(RES / "alternativas_rutas.csv", index=False)
    return df


def resumen_rutas(G):
    act, rec = rutas_actuales(G), rutas_recomendadas(G)
    filas = []
    for d in DESTINOS:
        a, r = act[d], rec[d]
        filas.append({"destino": d,
                      "ida_actual": a["t_ida"], "ida_dijkstra": r["t_ida"],
                      "reg_actual": a["t_regreso"], "reg_dijkstra": r["t_regreso"],
                      "ciclo_actual": a["t_ida"] + a["t_regreso"],
                      "ciclo_dijkstra": r["t_ida"] + r["t_regreso"],
                      "ruta_ida_dijkstra": fmt(r["ida"]), "ruta_reg_dijkstra": fmt(r["regreso"])})
    df = pd.DataFrame(filas)
    df["ahorro_ciclo"] = df.ciclo_actual - df.ciclo_dijkstra
    df["ahorro_pct"] = 100 * df.ahorro_ciclo / df.ciclo_actual
    df.to_csv(RES / "resumen_rutas.csv", index=False)
    return df


def comparar_algoritmos(G, reps=300):
    filas = []
    for d in DESTINOS:
        for o, t in ((BODEGA, d), (d, BODEGA)):
            def medir(fn):
                t0 = time.perf_counter()
                for _ in range(reps):
                    out = fn()
                return out, (time.perf_counter() - t0) / reps * 1e6
            (dist, prev, exp_d), us_d = medir(lambda: dijkstra(G, o, t))
            (cam_a, c_a, exp_a), us_a = medir(lambda: a_estrella(G, o, t))
            (dist_b, prev_b, rel_b), us_b = medir(lambda: bellman_ford(G, o))
            D, us_f = medir(lambda: floyd_warshall(G)) if (o, t) == (BODEGA, DESTINOS[0]) else (None, math.nan)
            filas.append({"par": f"{o}→{t}", "costo_dijkstra": dist[t], "costo_astar": c_a,
                          "costo_bellman_ford": dist_b[t],
                          "nodos_expandidos_dijkstra": exp_d, "nodos_expandidos_astar": exp_a,
                          "relajaciones_bf": rel_b,
                          "us_dijkstra": us_d, "us_astar": us_a, "us_bellman_ford": us_b,
                          "us_floyd_warshall_todos_pares": us_f})
    df = pd.DataFrame(filas)
    df.to_csv(RES / "comparacion_algoritmos.csv", index=False)
    D = floyd_warshall(G)
    assert all(abs(D[r.par.split("→")[0]][r.par.split("→")[1]] - r.costo_dijkstra) < 1e-9 for r in df.itertuples())
    return df


# --------------------------------------------------------------------------- #
def experimentos_base(G, vehiculos=VEHICULOS, descarga=0.0, cv=0.0, etiqueta="base"):
    filas = []
    for esc, tasa in ESCENARIOS.items():
        for s in SEMILLAS:
            pedidos = generar_pedidos(tasa, HORAS, s, cv_trafico=cv)   # mismos pedidos (CRN)
            for pol in ("Actual", "Dijkstra"):
                m = correr(G, pol, pedidos, vehiculos, HORAS, descarga).metricas()
                filas.append({"escenario": esc, "tasa": tasa, "semilla": s, "politica": pol, **m})
    df = pd.DataFrame(filas)
    df.to_csv(RES / f"experimentos_{etiqueta}.csv", index=False)
    return df


METRICAS = [("pedidos", "Pedidos recibidos"), ("puntuales", "Entregas puntuales"),
            ("pct_puntual", "% puntual"), ("espera_prom", "Espera prom. (min)"),
            ("sistema_prom", "Solicitud→entrega prom. (min)"),
            ("utilizacion", "Utilización (%)"), ("tiempo_extra", "Tiempo extra (min)")]


def resumen_experimentos(df, etiqueta="base"):
    filas = []
    for esc in ESCENARIOS:
        for pol in ("Actual", "Dijkstra"):
            sub = df[(df.escenario == esc) & (df.politica == pol)]
            fila = {"escenario": esc, "politica": pol}
            for m, _ in METRICAS:
                x = sub[m].to_numpy()
                h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
                fila[m] = x.mean()
                fila[m + "_ic95"] = h
            filas.append(fila)
    res = pd.DataFrame(filas)
    res.to_csv(RES / f"resumen_{etiqueta}.csv", index=False)
    # diferencias pareadas (Dijkstra - Actual), IC 95 % con t de Student
    dif = []
    for esc in ESCENARIOS:
        a = df[(df.escenario == esc) & (df.politica == "Actual")].set_index("semilla")
        b = df[(df.escenario == esc) & (df.politica == "Dijkstra")].set_index("semilla")
        for m, nombre in METRICAS[1:]:
            d = (b[m] - a[m]).to_numpy()
            h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
            p = stats.ttest_rel(b[m], a[m]).pvalue if d.std() > 0 else float("nan")
            dif.append({"escenario": esc, "metrica": nombre, "dif_media": d.mean(),
                        "ic95_inf": d.mean() - h, "ic95_sup": d.mean() + h, "p_valor": p})
    dif = pd.DataFrame(dif)
    dif.to_csv(RES / f"diferencias_pareadas_{etiqueta}.csv", index=False)
    return res, dif


def barrido_vehiculos(G, rango=range(4, 25), semillas=SEMILLAS):
    filas = []
    for esc, tasa in ESCENARIOS.items():
        lotes = {s: generar_pedidos(tasa, HORAS, s) for s in semillas}
        for c in rango:
            for pol in ("Actual", "Dijkstra"):
                ms = [correr(G, pol, lotes[s], c, HORAS).metricas() for s in semillas]
                m = pd.DataFrame(ms).mean()
                filas.append({"escenario": esc, "vehiculos": c, "politica": pol, **m.to_dict()})
    df = pd.DataFrame(filas)
    df.to_csv(RES / "barrido_vehiculos.csv", index=False)
    return df


def estudio_incidentes(G):
    casos = {
        "Red original": [],
        "Cierre N17→N25": [("cerrar", "N17", "N25")],
        "Cierre N03→N04": [("cerrar", "N03", "N04")],
        "Congestión N02→N18 (5→15 min)": [("tiempo", "N02", "N18", 15)],
        "Cierre N29→N31 y N27→N29": [("cerrar", "N29", "N31"), ("cerrar", "N27", "N29")],
    }
    filas_r, filas_s = [], []
    for nombre, cambios in casos.items():
        H = copiar(G)
        for c in cambios:
            if c[0] == "cerrar":
                H[c[1]].pop(c[2], None)
            else:
                H[c[1]][c[2]] = c[3]
        for pol in ("Actual", "Dijkstra"):
            P = politica(H, pol)
            for d in DESTINOS:
                filas_r.append({"caso": nombre, "politica": pol, "destino": d,
                                "ruta_ida": fmt(P[d]["ida"]), "t_ida": P[d]["t_ida"],
                                "ruta_regreso": fmt(P[d]["regreso"]), "t_regreso": P[d]["t_regreso"],
                                "desvio": P[d]["desvio"]})
            for esc, tasa in ESCENARIOS.items():
                ms = [correr(H, pol, generar_pedidos(tasa, HORAS, s), VEHICULOS, HORAS).metricas()
                      for s in SEMILLAS]
                m = pd.DataFrame(ms).mean()
                filas_s.append({"caso": nombre, "politica": pol, "escenario": esc,
                                **{k: m[k] for k, _ in METRICAS}})
    r, s = pd.DataFrame(filas_r), pd.DataFrame(filas_s)
    r.to_csv(RES / "incidentes_rutas.csv", index=False)
    s.to_csv(RES / "incidentes_simulacion.csv", index=False)
    return r, s


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def figuras_rutas(G):
    rec = rutas_recomendadas(G)
    for d in DESTINOS:
        act = RUTAS_ACTUALES[d]
        dibujar_grafo(G, resaltar={
            f"Actual ida/regreso ({tiempo_ruta(G, act):g} + {tiempo_ruta(G, act[::-1]):g} min)": (act, COL_ACT),
            f"Dijkstra ida ({rec[d]['t_ida']:g} min)": (rec[d]["ida"], COL_DIJ),
            f"Dijkstra regreso ({rec[d]['t_regreso']:g} min)": (rec[d]["regreso"], "#6c3fb5"),
        }, titulo=f"Destino {d}: ruta actual vs. ruta de Dijkstra", ruta_archivo=FIG / f"ruta_{d}.png",
            figsize=(13, 8.5))
        plt.close("all")


def figura_ciclos(rr):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(rr))
    w = 0.38
    ax.bar(x - w / 2, rr.ida_actual, w, color=COL_ACT, label="Actual - ida")
    ax.bar(x - w / 2, rr.reg_actual, w, bottom=rr.ida_actual, color=COL_ACT, alpha=0.45, label="Actual - regreso")
    ax.bar(x + w / 2, rr.ida_dijkstra, w, color=COL_DIJ, label="Dijkstra - ida")
    ax.bar(x + w / 2, rr.reg_dijkstra, w, bottom=rr.ida_dijkstra, color=COL_DIJ, alpha=0.45, label="Dijkstra - regreso")
    ax.axhline(75, ls="--", color="#555", lw=1)
    ax.text(len(rr) - 0.5, 77, "límite de puntualidad (75 min, solo ida)", ha="right", fontsize=8, color="#555")
    for i, r in rr.iterrows():
        ax.text(i - w / 2, r.ciclo_actual + 2, f"{r.ciclo_actual:g}", ha="center", fontsize=9)
        ax.text(i + w / 2, r.ciclo_dijkstra + 2, f"{r.ciclo_dijkstra:g}", ha="center", fontsize=9)
    ax.set_xticks(x, rr.destino)
    ax.set_ylabel("minutos")
    ax.set_title("Tiempo de ciclo por destino (ida + regreso)")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "ciclos_por_destino.png", dpi=160)
    plt.close(fig)


def figura_boxplots(df, etiqueta="base"):
    mets = [("pct_puntual", "% entregas puntuales"), ("espera_prom", "Espera promedio (min)"),
            ("sistema_prom", "Solicitud → entrega (min)"), ("utilizacion", "Utilización vehículos (%)"),
            ("tiempo_extra", "Tiempo extra después de 8 h (min)"), ("puntuales", "Entregas puntuales (n)")]
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, (m, tit) in zip(axs.flat, mets):
        datos, pos, cols = [], [], []
        for i, esc in enumerate(ESCENARIOS):
            for j, pol in enumerate(("Actual", "Dijkstra")):
                datos.append(df[(df.escenario == esc) & (df.politica == pol)][m])
                pos.append(i * 3 + j)
                cols.append(COL_ACT if pol == "Actual" else COL_DIJ)
        bp = ax.boxplot(datos, positions=pos, widths=0.75, patch_artist=True, medianprops=dict(color="black"))
        for b, c in zip(bp["boxes"], cols):
            b.set_facecolor(c)
            b.set_alpha(0.75)
        ax.set_xticks([0.5, 3.5, 6.5], list(ESCENARIOS))
        ax.set_title(tit, fontsize=10)
    axs[0, 0].plot([], [], "s", color=COL_ACT, label="Rutas actuales")
    axs[0, 0].plot([], [], "s", color=COL_DIJ, label="Rutas Dijkstra")
    axs[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Resultados por escenario ({len(SEMILLAS)} réplicas pareadas, {VEHICULOS} vehículos)")
    fig.tight_layout()
    fig.savefig(FIG / f"boxplots_{etiqueta}.png", dpi=160)
    plt.close(fig)


def figura_cola(G, semilla=1):
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), sharey=False)
    for ax, (esc, tasa) in zip(axs, ESCENARIOS.items()):
        ped = generar_pedidos(tasa, HORAS, semilla)
        for pol, c in (("Actual", COL_ACT), ("Dijkstra", COL_DIJ)):
            r = correr(G, pol, ped, VEHICULOS, HORAS)
            t, q = zip(*r.cola_t)
            ax.step(np.array(t) / 60, q, where="post", color=c, label=pol)
        ax.axvline(HORAS, ls="--", color="#555", lw=1)
        ax.set_title(f"{esc} (λ = {tasa:g}/h)")
        ax.set_xlabel("hora de operación")
    axs[0].set_ylabel("pedidos en cola")
    axs[0].legend(frameon=False)
    fig.suptitle(f"Evolución de la cola (semilla {semilla})")
    fig.tight_layout()
    fig.savefig(FIG / "cola_tiempo.png", dpi=160)
    plt.close(fig)


def figura_barrido(bv):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (m, tit) in zip(axs, [("pct_puntual", "% entregas puntuales"),
                                  ("espera_prom", "Espera promedio (min)"),
                                  ("tiempo_extra", "Tiempo extra (min)")]):
        for esc, ls in zip(ESCENARIOS, ("-", "--", ":")):
            for pol, c in (("Actual", COL_ACT), ("Dijkstra", COL_DIJ)):
                s = bv[(bv.escenario == esc) & (bv.politica == pol)]
                ax.plot(s.vehiculos, s[m], ls=ls, color=c, marker="o", ms=3, label=f"{esc} - {pol}")
        ax.axvline(8, color="#888", lw=0.8)
        ax.set_xlabel("número de vehículos")
        ax.set_title(tit)
    axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "barrido_vehiculos.png", dpi=160)
    plt.close(fig)


def capacidad_teorica(rr):
    """Carga ofrecida rho = lambda*E[ciclo]/(60*c). Si rho >= 1 la cola crece sin límite."""
    filas = []
    ca, cd = rr.ciclo_actual.mean(), rr.ciclo_dijkstra.mean()
    for esc, tasa in ESCENARIOS.items():
        filas.append({"escenario": esc, "tasa": tasa,
                      "ciclo_medio_actual": ca, "ciclo_medio_dijkstra": cd,
                      "rho_actual_8veh": tasa * ca / (60 * VEHICULOS),
                      "rho_dijkstra_8veh": tasa * cd / (60 * VEHICULOS),
                      "veh_min_actual": math.ceil(tasa * ca / 60 + 1e-9),
                      "veh_min_dijkstra": math.ceil(tasa * cd / 60 + 1e-9)})
    df = pd.DataFrame(filas)
    df.to_csv(RES / "capacidad_teorica.csv", index=False)
    return df


def main():
    G = cargar_grafo(AQUI / "matriz.csv")
    sim, asim, uno = clasificar_aristas(G)
    print(f"Nodos: {len(G)}  arcos dirigidos: {sum(len(v) for v in G.values())}")
    print(f"  tramos doble sentido mismo tiempo: {len(sim)}, distinto tiempo: {len(asim)}, un sentido: {len(uno)}")
    dibujar_grafo(G, titulo="Red de Logística del Valle (tiempos en minutos)", ruta_archivo=FIG / "grafo.png")
    plt.close("all")
    print(validar_actuales(G).to_string())
    print(tabla_alternativas(G).to_string())
    rr = resumen_rutas(G)
    print(rr.to_string())
    print(comparar_algoritmos(G).round(2).to_string())
    print(capacidad_teorica(rr).round(3).to_string())
    figuras_rutas(G)
    figura_ciclos(rr)
    df = experimentos_base(G)
    res, dif = resumen_experimentos(df)
    print(res.round(2).to_string())
    print(dif.round(3).to_string())
    figura_boxplots(df)
    figura_cola(G)
    df2 = experimentos_base(G, descarga=5.0, cv=0.15, etiqueta="sensibilidad")
    res2, dif2 = resumen_experimentos(df2, "sensibilidad")
    print(res2.round(2).to_string())
    bv = barrido_vehiculos(G)
    figura_barrido(bv)
    r, s = estudio_incidentes(G)
    print(r.to_string())
    print(s.round(1).to_string())


if __name__ == "__main__":
    main()
