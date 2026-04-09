"""
Visualización de la simulación de la ecuación de onda 3D.

Qué hace este script:
1. Lee el archivo data/wave.dat generado por el programa en C.
2. Reconstruye cada snapshot 2D del corte central.
3. Crea una animación con matplotlib.
4. Guarda esa animación como python/wave.gif.
5. También la muestra en pantalla.

Formato esperado del archivo data/wave.dat:
- Cada línea con datos tiene:
      i  j  valor
- Una línea en blanco separa un frame del siguiente.

Cada frame representa un corte 2D del volumen 3D,
tomado en el plano central z = NZ/2.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ============================================================
# 1. PARÁMETROS DEL PROBLEMA
# ============================================================
#
# Deben coincidir con las dimensiones usadas en el código C.
# Si cambias NX o NY en C, también debes cambiarlo aquí.
#
NX = 64
NY = 64

# Archivo de entrada generado por el código C
INPUT_FILE = "data/wave.dat"

# Archivo de salida para guardar la animación
OUTPUT_GIF = "python/wave.gif"


# ============================================================
# 2. LECTURA DE LOS DATOS
# ============================================================
#
# Esta función reconstruye los frames a partir de wave.dat.
#
# Idea:
# - Cada bloque de líneas corresponde a un frame.
# - Cada línea trae (i, j, u).
# - Una línea vacía indica que termina un frame.
#
def load_wave_data(filename, nx, ny):
    """
    Lee el archivo generado por la simulación en C y devuelve
    un arreglo de frames 2D.

    Parámetros
    ----------
    filename : str
        Ruta al archivo de datos.
    nx, ny : int
        Dimensiones del frame 2D.

    Retorna
    -------
    frames : np.ndarray
        Arreglo de forma (n_frames, nx, ny).
    """
    frames = []
    current = []

    # Abrimos el archivo de texto para leerlo línea por línea
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # Si encontramos una línea vacía, significa que terminó un frame
            if line == "":
                if current:
                    # Creamos un frame vacío
                    frame = np.zeros((nx, ny))

                    # Insertamos cada valor en su posición (i,j)
                    for i, j, u in current:
                        frame[i, j] = u

                    # Guardamos el frame completo
                    frames.append(frame)

                    # Reiniciamos el acumulador para el siguiente frame
                    current = []
                continue

            # Si la línea tiene contenido, esperamos tres columnas:
            # i, j y valor de la amplitud
            parts = line.split()

            if len(parts) == 3:
                i, j, u = parts
                current.append((int(i), int(j), float(u)))

    # Si el archivo no termina en línea vacía, igual guardamos el último frame
    if current:
        frame = np.zeros((nx, ny))
        for i, j, u in current:
            frame[i, j] = u
        frames.append(frame)

    return np.array(frames)


# ============================================================
# 3. CARGA DE FRAMES
# ============================================================
#
# Leemos el archivo producido por el programa en C.
#
frames = load_wave_data(INPUT_FILE, NX, NY)

# Verificación básica para detectar si el archivo está vacío
if len(frames) == 0:
    raise ValueError("No se encontraron frames en data/wave.dat")

# ============================================================
# 4. ESCALA DE COLOR
# ============================================================
#
# Fijamos vmin y vmax usando todos los frames.
# Esto es importante porque así la barra de color no cambia
# de escala en cada instante de la animación.
#
vmin = np.min(frames)
vmax = np.max(frames)

# ============================================================
# 5. CREACIÓN DE LA FIGURA
# ============================================================
#
# Creamos una figura y un eje donde mostraremos los frames.
#
fig, ax = plt.subplots(figsize=(6, 5))

# imshow representa el frame 2D como imagen
#
# origin="lower" hace que el origen aparezca abajo a la izquierda,
# que suele ser más natural para interpretar la grilla.
#
# cmap define el mapa de color.
#
# vmin y vmax fijan la escala global de colores.
#
im = ax.imshow(
    frames[0],
    origin="lower",
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
    animated=True
)

# Etiquetas y título inicial
ax.set_xlabel("j")
ax.set_ylabel("i")
ax.set_title("Ecuación de onda 3D: corte central")

# Barra de color para indicar el valor de u
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("u")

# ============================================================
# 6. FUNCIÓN DE ACTUALIZACIÓN DE LA ANIMACIÓN
# ============================================================
#
# Esta función recibe el índice del frame y actualiza la imagen.
#
def update(frame_index):
    """
    Actualiza la imagen para mostrar el frame correspondiente.
    """
    im.set_array(frames[frame_index])
    ax.set_title(f"Ecuación de onda 3D: corte central | frame {frame_index}")
    return [im]


# ============================================================
# 7. CREACIÓN DE LA ANIMACIÓN
# ============================================================
#
# FuncAnimation llama repetidamente a update(...)
# para construir la animación.
#
ani = FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=200,
    blit=True
)

# ============================================================
# 8. GUARDAR COMO GIF
# ============================================================
#
# Guardamos la animación en un GIF para poder verla fácilmente
# y también subirla a GitHub.
#
ani.save(OUTPUT_GIF, writer=PillowWriter(fps=5))
print(f"Animación guardada en: {OUTPUT_GIF}")

# ============================================================
# 9. MOSTRAR EN PANTALLA
# ============================================================
#
# Esto abre una ventana con la animación si estás trabajando
# en un entorno gráfico local.
#
plt.show()
