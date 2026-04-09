import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

NX = 64
NY = 64

def load_wave_data(filename, nx, ny):
    frames = []
    current = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                if current:
                    frame = np.zeros((nx, ny))
                    for i, j, u in current:
                        frame[int(i), int(j)] = float(u)
                    frames.append(frame)
                    current = []
                continue

            parts = line.split()
            if len(parts) == 3:
                i, j, u = parts
                current.append((int(i), int(j), float(u)))

    if current:
        frame = np.zeros((nx, ny))
        for i, j, u in current:
            frame[int(i), int(j)] = float(u)
        frames.append(frame)

    return np.array(frames)

frames = load_wave_data("wave.dat", NX, NY)

fig, ax = plt.subplots()
im = ax.imshow(
    frames[0],
    origin="lower",
    cmap="viridis",
    animated=True
)

ax.set_title("Ecuación de onda 3D: corte central")
ax.set_xlabel("j")
ax.set_ylabel("i")
fig.colorbar(im, ax=ax, label="u")

def update(frame_index):
    im.set_array(frames[frame_index])
    ax.set_title(f"Ecuación de onda 3D: corte central | frame {frame_index}")
    return [im]

ani = FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=200,
    blit=True
)

plt.show()
