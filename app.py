"""
app.py - Aplicación configurable (Streamlit) para Logística del Valle.
Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from red import (BODEGA, DESTINOS, RUTAS_ACTUALES, cargar_grafo, copiar, k_rutas, tiempo_ruta)
from simulacion import ESCENARIOS, UMBRAL_PUNTUAL, correr, generar_pedidos, politica
from visual import dibujar_grafo

AQUI = Path(__file__).parent
COL = {"Actual": "#c0392b", "Dijkstra": "#1f6fb2"}

st.set_page_config(page_title="Logística del Valle - Simulador", layout="wide")


@st.cache_data
def grafo_base():
    return cargar_grafo(AQUI / "matriz.csv")


def tabla_aristas(G):
    return pd.DataFrame([{"origen": u, "destino": v, "minutos": w, "cerrado": False}
                         for u, vs in G.items() for v, w in vs.items()])


def fmt(r):
    return " → ".join(r) if r else "(inalcanzable)"


# --------------------------------------------------------------------------- #
# Barra lateral: parámetros
# --------------------------------------------------------------------------- #
st.sidebar.header("Parámetros de operación")
esc = st.sidebar.selectbox("Escenario", list(ESCENARIOS) + ["Personalizado"])
tasa_def = ESCENARIOS.get(esc, 4.0)
tasa = st.sidebar.number_input("Pedidos por hora (promedio λ)", 0.5, 40.0, float(tasa_def), 0.5,
                               key=f"tasa_{esc}", disabled=esc != "Personalizado",
                               help="En los escenarios predefinidos se fija según el caso; "
                                    "elige 'Personalizado' para cambiarlo.")
vehiculos = st.sidebar.number_input("Número de vehículos", 1, 60, 8, 1)
horas = st.sidebar.number_input("Duración de la recepción de pedidos (h)", 0.5, 24.0, 8.0, 0.5)
semilla = st.sidebar.number_input("Semilla aleatoria", 0, 10_000_000, 1, 1)
st.sidebar.subheader("Supuestos opcionales")
descarga = st.sidebar.number_input("Tiempo de entrega/descarga en destino (min)", 0.0, 60.0, 0.0, 1.0)
cv = st.sidebar.slider("Variabilidad del tráfico (CV del factor lognormal)", 0.0, 0.5, 0.0, 0.05,
                       help="0 = tiempos exactos de la matriz. Con CV>0 cada viaje se multiplica por un "
                            "factor aleatorio de media 1; ambas políticas reciben los mismos factores.")

# --------------------------------------------------------------------------- #
# Red editable (incidentes viales)
# --------------------------------------------------------------------------- #
G0 = grafo_base()
if "aristas" not in st.session_state:
    st.session_state.aristas = tabla_aristas(G0)

st.title("Logística del Valle - Simulador de rutas y flota")
st.caption("Bodega: N01. Destinos: N08, N19, N34 y N40. Entrega puntual: 75 min o menos desde la solicitud.")

tab_red, tab_sim, tab_exp, tab_flota = st.tabs(
    ["Red y rutas", "Simulación (1 corrida)", "Experimentos", "Tamaño de flota"])

with tab_red:
    st.subheader("Incidentes viales: modificar o cerrar tramos")
    st.write("Edita la columna **minutos** para simular congestión o marca **cerrado** para quitar el tramo. "
             "Al cambiar la red, Dijkstra recalcula las rutas recomendadas al instante; en la política "
             "*Actual*, el conductor sigue su hoja hasta el punto anterior al tramo cerrado y desde ahí "
             "toma el desvío más corto.")
    c1, c2 = st.columns([3, 1])
    with c2:
        filtro = st.text_input("Filtrar por nodo (p. ej. N17)", "")
        if st.button("Restablecer red original"):
            st.session_state.aristas = tabla_aristas(G0)
            st.rerun()
    df_ar = st.session_state.aristas
    vista = df_ar if not filtro else df_ar[(df_ar.origen == filtro.upper()) | (df_ar.destino == filtro.upper())]
    with c1:
        editado = st.data_editor(vista, hide_index=True, width="stretch", height=260,
                                 disabled=["origen", "destino"],
                                 column_config={"minutos": st.column_config.NumberColumn(min_value=0.1, step=1.0)})
    st.session_state.aristas.loc[editado.index, ["minutos", "cerrado"]] = editado[["minutos", "cerrado"]]

df_ar = st.session_state.aristas
G = {n: {} for n in G0}
for r in df_ar.itertuples():
    if not r.cerrado:
        G[r.origen][r.destino] = float(r.minutos)
cambios = df_ar.merge(tabla_aristas(G0), on=["origen", "destino"], suffixes=("", "_orig"))
cambios = cambios[(cambios.minutos != cambios.minutos_orig) | cambios.cerrado]

try:
    POL = {p: politica(G, p) for p in ("Actual", "Dijkstra")}
    red_ok = all(not math.isinf(POL[p][d][k]) for p in POL for d in DESTINOS for k in ("t_ida", "t_regreso"))
except Exception:
    red_ok = False

with tab_red:
    if len(cambios):
        st.info("Cambios activos: " + "; ".join(
            f"{r.origen}→{r.destino} " + ("CERRADO" if r.cerrado else f"{r.minutos_orig:g}→{r.minutos:g} min")
            for r in cambios.itertuples()))
    if not red_ok:
        st.error("Con los cambios actuales algún destino o el regreso a N01 queda inalcanzable.")
    st.subheader("Rutas por destino")
    filas = []
    for d in DESTINOS:
        for p in ("Actual", "Dijkstra"):
            x = POL[p][d]
            filas.append({"Destino": d, "Política": p, "Ruta ida": fmt(x["ida"]), "Ida (min)": x["t_ida"],
                          "Ruta regreso": fmt(x["regreso"]), "Regreso (min)": x["t_regreso"],
                          "Ciclo (min)": x["t_ida"] + x["t_regreso"],
                          "¿Puntual sin espera?": "Sí" if x["t_ida"] + descarga <= UMBRAL_PUNTUAL else "No",
                          "Desvío por incidente": "Sí" if x.get("desvio") else ""})
    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")

    d_sel = st.selectbox("Ver en el mapa", DESTINOS)
    with st.expander("Rutas alternativas (Yen, k = 4)"):
        alt = []
        for sentido, (o, t) in (("Ida", (BODEGA, d_sel)), ("Regreso", (d_sel, BODEGA))):
            for i, (c, r) in enumerate(k_rutas(G, o, t, 4)):
                alt.append({"Sentido": sentido, "#": i + 1, "Ruta": fmt(r), "Minutos": c})
        st.dataframe(pd.DataFrame(alt), hide_index=True, width="stretch")
    res = {}
    a = POL["Actual"][d_sel]
    if a["ida"]:
        res[f"Actual ida ({a['t_ida']:g} min)"] = (a["ida"], "#c0392b")
    if a["regreso"] and a["regreso"] != a["ida"][::-1]:
        res[f"Actual regreso ({a['t_regreso']:g} min)"] = (a["regreso"], "#e67e22")
    b = POL["Dijkstra"][d_sel]
    if b["ida"]:
        res[f"Dijkstra ida ({b['t_ida']:g} min)"] = (b["ida"], "#1f6fb2")
    if b["regreso"]:
        res[f"Dijkstra regreso ({b['t_regreso']:g} min)"] = (b["regreso"], "#6c3fb5")
    cerr = [(r.origen, r.destino) for r in df_ar.itertuples() if r.cerrado]
    fig = dibujar_grafo(G, resaltar=res, cerradas=cerr, figsize=(13, 8.5),
                        titulo=f"Destino {d_sel}")
    st.pyplot(fig)
    plt.close(fig)

if not red_ok:
    st.stop()


def gantt(r, titulo, color):
    fig, ax = plt.subplots(figsize=(11, 0.35 * r.n_vehiculos + 1.2))
    for p in r.pedidos:
        ax.barh(p.vehiculo, p.entrega - p.salida, left=p.salida / 1, color=color, alpha=0.9, height=0.6)
        ax.barh(p.vehiculo, p.regreso - p.entrega, left=p.entrega, color=color, alpha=0.4, height=0.6)
    ax.axvline(horas * 60, ls="--", color="#555")
    ax.set_yticks(range(r.n_vehiculos), [f"V{i+1}" for i in range(r.n_vehiculos)])
    ax.set_xlabel("minuto (barra oscura = ida, clara = regreso)")
    ax.set_title(titulo)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


ETQ = {"pedidos": "Pedidos recibidos", "puntuales": "Entregas puntuales", "pct_puntual": "% puntual",
       "espera_prom": "Espera promedio (min)", "espera_max": "Espera máxima (min)",
       "sistema_prom": "Solicitud→entrega prom. (min)", "sistema_p90": "Solicitud→entrega P90 (min)",
       "utilizacion": "Utilización vehículos (%)", "util_jornada": "Utilización en la jornada (%)",
       "fin_min": "Último regreso a N01 (min)", "tiempo_extra": "Tiempo extra tras la jornada (min)",
       "extra_entregas": "Última entrega después de la jornada (min)", "cola_max": "Cola máxima"}

with tab_sim:
    pedidos = generar_pedidos(tasa, horas, int(semilla), cv_trafico=cv)
    R = {p: correr(G, p, pedidos, int(vehiculos), horas, descarga, bitacora=True) for p in POL}
    st.write(f"**{len(pedidos)} pedidos** generados con λ = {tasa:g}/h durante {horas:g} h "
             f"(semilla {semilla}). Ambas políticas atienden exactamente los mismos pedidos.")
    M = pd.DataFrame({p: R[p].metricas() for p in R}).rename(index=ETQ)
    M["Diferencia (Dijkstra − Actual)"] = M["Dijkstra"] - M["Actual"]
    st.dataframe(M.round(2), width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        for p, r in R.items():
            t, q = zip(*r.cola_t)
            ax.step(np.array(t) / 60, q, where="post", color=COL[p], label=p)
        ax.axvline(horas, ls="--", color="#555")
        ax.set_xlabel("hora")
        ax.set_ylabel("pedidos en cola")
        ax.legend()
        ax.set_title("Cola en el tiempo")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        for p, r in R.items():
            ax.hist([x.tiempo_sistema for x in r.pedidos], bins=25, alpha=0.6, color=COL[p], label=p)
        ax.axvline(UMBRAL_PUNTUAL, ls="--", color="#555")
        ax.set_xlabel("minutos desde la solicitud hasta la entrega")
        ax.legend()
        ax.set_title("Distribución del tiempo de entrega")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    for p, r in R.items():
        st.pyplot(gantt(r, f"Uso de vehículos - política {p}", COL[p]))
        plt.close("all")
    with st.expander("Detalle por pedido y bitácora de eventos"):
        p_sel = st.radio("Política", list(R), horizontal=True)
        r = R[p_sel]
        st.dataframe(pd.DataFrame([{"pedido": x.id, "llegada": x.llegada, "destino": x.destino,
                                    "vehículo": f"V{x.vehiculo+1}", "salida": x.salida, "espera": x.espera,
                                    "entrega": x.entrega, "regreso": x.regreso,
                                    "solicitud→entrega": x.tiempo_sistema, "puntual": x.puntual}
                                   for x in r.pedidos]).round(2), hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame(r.eventos, columns=["minuto", "evento", "pedido", "vehículo"]).round(2),
                     hide_index=True, width="stretch", height=250)

with tab_exp:
    st.write("Réplicas pareadas: para cada semilla se generan los pedidos una sola vez y se simulan "
             "con ambas políticas (números aleatorios comunes).")
    c1, c2 = st.columns(2)
    reps = c1.number_input("Repeticiones por escenario", 5, 200, 10, 1)
    escs = c2.multiselect("Escenarios", list(ESCENARIOS) + ["Configuración actual"],
                          default=list(ESCENARIOS))
    if st.button("Ejecutar experimentos", type="primary"):
        filas = []
        for e in escs:
            lam = tasa if e == "Configuración actual" else ESCENARIOS[e]
            for s in range(int(semilla), int(semilla) + int(reps)):
                ped = generar_pedidos(lam, horas, s, cv_trafico=cv)
                for p in POL:
                    m = correr(G, p, ped, int(vehiculos), horas, descarga).metricas()
                    filas.append({"escenario": e, "semilla": s, "política": p, **m})
        st.session_state.exp = pd.DataFrame(filas)
    if "exp" in st.session_state:
        df = st.session_state.exp
        cols = ["pedidos", "puntuales", "pct_puntual", "espera_prom", "sistema_prom", "utilizacion", "tiempo_extra"]
        tabla = df.groupby(["escenario", "política"], sort=False)[cols].agg(["mean", "std"]).round(2)
        st.dataframe(tabla, width="stretch")
        fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
        for ax, m in zip(axs, ["pct_puntual", "espera_prom", "utilizacion", "tiempo_extra"]):
            es = list(dict.fromkeys(df.escenario))
            data, pos, cs = [], [], []
            for i, e in enumerate(es):
                for j, p in enumerate(POL):
                    data.append(df[(df.escenario == e) & (df["política"] == p)][m])
                    pos.append(i * 3 + j)
                    cs.append(COL[p])
            bp = ax.boxplot(data, positions=pos, patch_artist=True, widths=0.7)
            for b, c in zip(bp["boxes"], cs):
                b.set_facecolor(c)
                b.set_alpha(0.7)
            ax.set_xticks([i * 3 + 0.5 for i in range(len(es))], es, fontsize=8)
            ax.set_title(ETQ[m], fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.download_button("Descargar resultados (CSV)", df.to_csv(index=False).encode(), "experimentos.csv")

with tab_flota:
    st.write("Promedio de las réplicas para distintos tamaños de flota (red y parámetros de la barra lateral).")
    c1, c2, c3 = st.columns(3)
    vmin = c1.number_input("Desde", 1, 60, 4)
    vmax = c2.number_input("Hasta", 1, 80, 20)
    reps_f = c3.number_input("Réplicas", 1, 100, 10, key="reps_f")
    if st.button("Calcular barrido"):
        lotes = [generar_pedidos(tasa, horas, s, cv_trafico=cv) for s in range(int(semilla), int(semilla) + int(reps_f))]
        filas = []
        for c in range(int(vmin), int(vmax) + 1):
            for p in POL:
                m = pd.DataFrame([correr(G, p, l, c, horas, descarga).metricas() for l in lotes]).mean()
                filas.append({"vehículos": c, "política": p, **m.to_dict()})
        st.session_state.flota = pd.DataFrame(filas)
    if "flota" in st.session_state:
        bv = st.session_state.flota
        fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
        for ax, m in zip(axs, ["pct_puntual", "espera_prom", "utilizacion"]):
            for p in POL:
                s = bv[bv["política"] == p]
                ax.plot(s["vehículos"], s[m], marker="o", color=COL[p], label=p)
            ax.set_title(ETQ[m])
            ax.set_xlabel("vehículos")
        axs[0].legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        ciclo = {p: np.mean([POL[p][d]["t_ida"] + POL[p][d]["t_regreso"] for d in DESTINOS]) + descarga for p in POL}
        st.write("Carga ofrecida: rho = lambda * E[ciclo] / (60 * c). Si rho >= 1 la cola crece durante toda la jornada.")
        st.dataframe(pd.DataFrame([{"política": p, "ciclo medio (min)": ciclo[p],
                                    "vehículos mínimos para ρ<1": math.floor(tasa * ciclo[p] / 60) + 1,
                                    f"ρ con {vehiculos} vehículos": tasa * ciclo[p] / (60 * vehiculos)}
                                   for p in POL]).round(3), hide_index=True)
        st.dataframe(bv.round(2), hide_index=True, width="stretch")
