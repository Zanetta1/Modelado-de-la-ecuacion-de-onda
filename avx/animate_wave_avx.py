import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Deben coincidir con los usados en el código C.
#
NX = 64
NY = 64

INPUT_FILE = "../data/wave_avx.dat"
OUTPUT_GIF = "../python/wave_avx.gif"

# ============================================================
# FUNCIÓN PARA LEER LOS FRAMES
# ============================================================

def load_wave_data(filename, nx, ny):
	
    frames = []
    current = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # Línea vacía: terminó un frame
            if line == "":
                if current:
                    frame = np.zeros((nx, ny))

                    for i, j, u in current:
                        frame[i, j] = u

                    frames.append(frame)
                    current = []
                continue

            parts = line.split()

            if len(parts) == 3:
                i, j, u = parts
                current.append((int(i), int(j), float(u)))

    # Por si el archivo no termina con línea vacía
    if current:
        frame = np.zeros((nx, ny))
        for i, j, u in current:
            frame[i, j] = u
        frames.append(frame)

    return np.array(frames)
    
# ============================================================
# 3. CARGA DE DATOS
# ============================================================
frames = load_wave_data(INPUT_FILE, NX, NY)

if len(frames) == 0:
    raise ValueError("No se encontraron frames en data/wave_avx.dat")


# ============================================================
# 4. ESCALA GLOBAL DE COLOR
# ============================================================
#
# Se fija usando todos los frames para que la barra de color
# no cambie en cada instante.
#
vmin = np.min(frames)
vmax = np.max(frames)

# Si por alguna razón todos los valores fueran iguales,
# ajustamos levemente el rango para evitar problemas visuales.
if vmin == vmax:
    vmin -= 1e-12
    vmax += 1e-12

# ============================================================
# 5. CREACIÓN DE FIGURA
# ============================================================
fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(
    frames[0],
    origin="lower",
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
    animated=True
)

ax.set_xlabel("j")
ax.set_ylabel("i")
ax.set_title("Ecuación de onda 3D: versión AVX")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("u")

# ============================================================
# 6. FUNCIÓN DE ACTUALIZACIÓN
# ============================================================
def update(frame_index):
    """
    Actualiza la imagen para mostrar un frame específico.
    """
    im.set_array(frames[frame_index])
    ax.set_title(f"Ecuación de onda 3D: versión AVX | frame {frame_index}")
    return [im]

# ============================================================
# 7. CREACIÓN DE LA ANIMACIÓN
# ============================================================
ani = FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=200,
    blit=True
)

# ============================================================
# 8. GUARDAR GIF
# ============================================================
ani.save(OUTPUT_GIF, writer=PillowWriter(fps=5))
print(f"Animación guardada en: {OUTPUT_GIF}")

# ============================================================
# 9. MOSTRAR EN PANTALLA
# ============================================================
plt.show()



