# Modelado de la ecuación de onda

Este proyecto resuelve numéricamente la ecuación de onda en tres dimensiones mediante diferencias finitas usando C, y luego visualiza la evolución de un corte central de la solución usando Python.

## Descripción general

El programa en C construye una grilla tridimensional uniforme, define una perturbación inicial tipo gaussiana y hace evolucionar la onda en el tiempo mediante un esquema explícito de diferencias finitas.

Durante la simulación se guarda un corte central del volumen en un archivo de datos. Luego, un script en Python lee esos datos y genera una animación en formato GIF para visualizar la evolución temporal de la onda.

## Objetivo del proyecto

El objetivo de este trabajo es:

- simular la propagación de una perturbación inicial en un medio 3D
- almacenar datos numéricos de la evolución temporal
- visualizar los resultados mediante una animación
- medir el tiempo de ejecución para comparar posteriormente con una versión paralelizada

## Estructura del proyecto

```text
wave-equation-project/
├── data/
│   └── wave.dat
├── python/
│   ├── animate_wave.py
│   └── wave.gif
├── src/
│   ├── wave.c
│   └── wave3d
