# Práctica 4 - Logística del Valle (CC2017 Modelación y Simulación)

Javier Lopez, 23415

## Archivos
| Archivo | Contenido |
|---|---|
| `matriz.csv` | Anexo A: matriz de adyacencia (minutos, fila = origen, columna = destino) |
| `red.py` | Carga del grafo dirigido, validación de rutas, Dijkstra, A*, Bellman-Ford, Floyd-Warshall, Yen (k rutas) |
| `simulacion.py` | Simulación de eventos discretos (llegadas Poisson, cola FIFO, flota, ida/entrega/regreso) con números aleatorios comunes |
| `visual.py` | Dibujo del grafo y de rutas resaltadas |
| `experimentos.py` | Genera todas las tablas (`resultados/*.csv`) y figuras (`figuras/*.png`) del informe |
| `app.py` | Aplicación configurable (Streamlit) |

## Informe
El informe está en `informe/Informe_Practica4_Logistica_del_Valle.pdf`.

## Uso
```bash
pip install -r requirements.txt
python experimentos.py        # reproduce tablas y figuras del informe
streamlit run app.py          # abre la aplicación en el navegador
```

En la aplicación se pueden cambiar: escenario (Normal 4/h, Semana Santa 6/h, Navidad 10/h o personalizado),
pedidos por hora, número de vehículos, duración de la recepción, semilla, tiempo de descarga y variabilidad
del tráfico. En la pestaña **Red y rutas** se edita el tiempo de cualquier tramo o se cierra; las rutas de
Dijkstra se recalculan al instante y la política actual toma un desvío desde el punto previo al cierre.
