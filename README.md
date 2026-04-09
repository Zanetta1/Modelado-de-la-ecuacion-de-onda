# Modelado de la ecuación de onda

Este proyecto resuelve numéricamente la ecuación de onda 3D mediante diferencias finitas usando C, y luego visualiza la evolución de un corte central de la solución usando Python.

## Ecuación modelada

La ecuación estudiada es:

\[
\frac{\partial^2 u}{\partial t^2}
=
c^2
\left(
\frac{\partial^2 u}{\partial x^2}
+
\frac{\partial^2 u}{\partial y^2}
+
\frac{\partial^2 u}{\partial z^2}
\right)
\]

donde:

- \(u(x,y,z,t)\) es la amplitud de la onda
- \(c\) es la velocidad de propagación

## Objetivo del proyecto

El objetivo de este trabajo es:

- simular la propagación de una perturbación inicial en un medio 3D
- almacenar datos numéricos de la evolución temporal
- visualizar los resultados mediante una animación
- medir el tiempo de ejecución para luego comparar con una futura versión paralelizada

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
