import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

plt.rcParams.update({
    "figure.figsize": (12, 9),
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "font.size": 10,
    "lines.linewidth": 2.0
})

def rosenbrock(x, y, a=1.0, b=100.0):
    return (a-x)**2 + b*(y-x**2)**2

def Lag(vars_, a, b, c):
    x, y, lam = vars_
    eq1 = -2*(a-x) - 4*b*x*(y-x**2) + 2*lam*x
    eq2 =  2*b*(y-x**2) + 2*lam*y
    eq3 = x**2 + y**2 - c**2
    return [eq1, eq2, eq3]

# ======================
# Parametros
a, b = 1.0, 100.0
c = 2.0

# logspace
niveles = np.logspace(-4, 3, 40)

# Malla 1 (banana clasica)
xmin, xmax = -3.0, 3.0
ymin, ymax = -3.0, 5.0
X= np.linspace(xmin, xmax, 700)
Y = np.linspace(ymin, ymax, 700)
XX, YY = np.meshgrid(X, Y)
ZZ = rosenbrock(XX, YY, a=a, b=b)



# Punto estacionario sin restriccion
x_sta, y_sta = a, a**2

# Solucion numerica restringida (para c=2)
x0 = np.array([1.0, 1.0, 0.0])
sol = fsolve(Lag, x0, args=(a, b, c))
x_sol, y_sol, lam_sol = sol

# Parametrizacion del circulo
t = np.linspace(0, 2*np.pi, 800)
xc = c*np.cos(t)
yc = c*np.sin(t)

# ======================
# Subplots: 2x2
fig, axs = plt.subplots(2, 2)

# (1) Banana + minimo sin restriccion
ax = axs[0, 0]
cf = ax.contourf(XX, YY, ZZ, levels=niveles, cmap="viridis")
ax.contour(XX, YY, ZZ, levels=niveles, colors="k", linewidths=0.35, alpha=0.6)
ax.scatter([x_sta], [y_sta], s=120, marker="*", label="minimo global (sin restr.)")
ax.set_title("Rosenbrock (a=1, b=100)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.legend(loc="upper left")
fig.colorbar(cf, ax=ax)

# (2) Banana + circulo (c=2)
ax = axs[0, 1]
cf = ax.contourf(XX, YY, ZZ, levels=niveles, cmap="viridis")
ax.contour(XX, YY, ZZ, levels=niveles, colors="k", linewidths=0.4, alpha=0.5)
ax.plot(xc, yc, label="restriccion: x^2+y^2=c^2 (c=2)")
ax.set_title("Rosenbrock + restriccion circular")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.legend(loc="upper left")
fig.colorbar(cf, ax=ax)

# (3) Optimo con restriccion (c=2) marcado
ax = axs[1, 0]
cf = ax.contourf(XX, YY, ZZ, levels=niveles, cmap="viridis")
ax.contour(XX, YY, ZZ, levels=niveles, colors="k", linewidths=0.4, alpha=0.5)
ax.plot(xc, yc, label="circulo (c=2)")
ax.scatter([x_sol], [y_sol], s=120, marker="X", label="estacionario restringido")
ax.set_title("Punto estacionario con restriccion (c=2)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.legend(loc="upper left")
fig.colorbar(cf, ax=ax)

# (4) Trayectoria de soluciones para varios c
ax = axs[1, 1]
cs = np.round(np.arange(0.0, 5.0 + 1e-12, 0.1), 1)
soluciones = []
exitos = []

seed = [
    np.array([1.0, 1.0, 0.0]),
    np.array([1.0, -1.0, 0.0]),
    np.array([-1.0, 1.0, 0.0]),
    np.array([-1.0, -1.0, 0.0]),
    np.array([0.5, 0.0, 0.0]),
    np.array([0.0, 0.5, 0.0]),
]
prev = np.array([1.0, 1.0, 0.0])

for c_i in cs:
    if c_i == 0.0:
        soluciones.append([0.0, 0.0, 0.0])
        exitos.append(True)
        prev = np.array([0.0, 0.0, 0.0])
        continue

    found = False
    intento = [prev] + seed
    for x0 in intento:
        sol_i, info, ier, msg = fsolve(Lag, x0, args=(a, b, c_i), full_output=True, maxfev=2000)
        if ier == 1:
            x_i, y_i, lam_i = sol_i
            if abs(x_i**2 + y_i**2 - c_i**2) < 1e-6:
                soluciones.append(sol_i)
                exitos.append(True)
                prev = sol_i
                found = True
                break

    if not found:
        soluciones.append([np.nan, np.nan, np.nan])
        exitos.append(False)

soluciones = np.array(soluciones)
ax.contourf(XX, YY, ZZ, levels=niveles, cmap="viridis", alpha=0.85)
ax.contour(XX, YY, ZZ, levels=niveles, colors="k", linewidths=0.35, alpha=0.4)
ax.plot(soluciones[:, 0], soluciones[:, 1], marker="o", markersize=3)

failed = np.sum(~np.array(exitos))
ax.set_title(f"Trayectoria soluciones vs c (fallas: {failed})")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal", adjustable="box")

plt.tight_layout()
plt.show()

