import numpy as np
import matplotlib. pyplot as plt
from scipy.optimize import root
def Rosen_N (x, a=1.0, b=100.0):
    x = np.asarray(x)
    return np.sum((a-x[:-1])**2 + b*(x[1:] - x[:-1]**2)**2)
a,b = 1.0, 100.0
grid = np.linspace(-3, 3, 500)
niveles = np.logspace (-3, 3, 40)

fig, axs = plt.subplots(1,3, figsize = (15,4), dpi = 120)

#====================================================================================
#Visualizacion
#Corte 1: x3 = 1, plano (x1,x2)
X1, X2 = np.meshgrid(grid, grid)
Z = np.zeros_like(X1)
x3prov = 1.0
for i in range(X1.shape[0]):
    Z[i,:] = [Rosen_N([X1[i,j], X2[i,j], x3prov], a=a, b=b) for j in range(X1.shape[1])]
axs[0].contourf(X1, X2, Z, levels=niveles, cmap="viridis")
axs[0].contour(X1, X2, Z, levels=niveles, colors="k", linewidths=0.25, alpha=0.35)
axs[0].set_title(r"Corte: $x_3=1$ en el plano $(x_1,x_2)$")
axs[0].set_xlabel(r"$x_1$")
axs[0].set_ylabel(r"$x_2$")

# corte 2: x2=1, plano (x1,x3)
X1, X3 = np.meshgrid(grid, grid)
Z = np.zeros_like(X1) 
x2fix = 1.0
for i in range(X1.shape[0]):
    Z[i,:] = [Rosen_N([X1[i,j], x2fix, X3[i,j]], a=a, b=b) for j in range(X1.shape[1])]
axs[1].contourf(X1, X3, Z, levels=niveles, cmap="viridis")
axs[1].contour(X1, X3, Z, levels=niveles, colors="k", linewidths=0.25, alpha=0.35)
axs[1].set_title(r"Corte: $x_2=1$ en el plano $(x_1,x_3)$")
axs[1].set_xlabel(r"$x_1$")
axs[1].set_ylabel(r"$x_3$")

#Corte 3: x1 = 1 plano (x2, x3)
X2, X3 = np.meshgrid(grid, grid)
Z = np.zeros_like(X2)
x1fix = 1.0
for i in range(X2.shape[0]):
    Z[i,:] = [Rosen_N([x1fix, X2[i,j], X3[i,j]], a=a, b=b) for j in range(X2.shape[1])]
axs[2].contourf(X2, X3, Z, levels=niveles, cmap="viridis")
axs[2].contour(X2, X3, Z, levels=niveles, colors="k", linewidths=0.25, alpha=0.35)
axs[2].set_title(r"Corte: $x_1=1$ en el plano $(x_2,x_3)$")
axs[2].set_xlabel(r"$x_2$")
axs[2].set_ylabel(r"$x_3$")

plt.tight_layout()
plt.show()

#================================================================================
#Resolucion numerica 
def grad_rosen_N(x, a=1.0, b=100.0):
    x = np.asarray(x)
    N = x.size
    g = np.zeros_like(x)
    #caso i = 0
    g[0] = -2*(a - x[0]) - 4*b*x[0]*(x[1] - x[0]**2)
    #caso 1<= i <= N-2
    for i in range (1, N-1):
        g[i] = -2*(a-x[i]) - 4*b*x[i]*(x[i+1] - x[i]**2) + 2*b*(x[i] - x[i-1]**2)
    #caso i = N-1
    g[N-1] = 2*b*(x[N-1]-x[N-2]**2)
    return g
a, b = 1.0, 100.0
N=100

x0 = np.ones(N) * 0.8

sol = root(lambda x: grad_rosen_N(x, a=a, b=b), x0, method = "hybr")
print("Converge: ", sol.success)
print("Norma del residuo ||grad||:", np.linalg.norm(sol.fun))
print("Primeras componentes: ", sol.x[:5])

#=================================================================================
#Con Restricciones ||x||^2 = c^2

def system_constrained(Z,a = 1.0, b =100.0, c=2.0):
    x= Z[:-1]
    lam = Z[-1]
    g = grad_rosen_N(x,a=a, b=b) + 2*lam*x
    h = np.dot(x,x) - c**2
    return np.hstack([g,h])
a,b = 1.0, 100.0
N=100
c = 2.0

x0 = (c/np.sqrt(N)) * np.ones(N)
lam0 = 0.0
z0 = np.hstack([x0, lam0])

sol = root (lambda z: system_constrained(z, a=a, b=b, c=c), z0, method="hybr")
x_sol = sol.x[:-1]
lam_sol = sol.x[-1]

print("Converge:", sol.success)
print("||grad f + 2lamx||:", np.linalg.norm(grad_rosen_N(x_sol, a=a, b=b) + 2*lam_sol*x_sol))
print("||x||^2:", np.dot(x_sol, x_sol), "  c^2:", c**2)
#============================================================================================0
#Secuencuas de c
cs = np.round(np.arange(0.0, 5.0 + 1e-12, 0.1), 1)
rng = np.random.default_rng(123)

def generate_seeds(N, c, n_seeds=6):
    seeds = []
    for _ in range(n_seeds):
        v = rng.normal(size=N)
        v /= np.linalg.norm(v)
        seeds.append(c * v)
    return seeds


N = 100
a, b = 1.0, 100.0

soluciones = []
exitos = []


prev_x = np.zeros(N)
prev_lam = 0.0

for c_i in cs:

    
    if c_i == 0.0:
        soluciones.append(np.hstack([np.zeros(N), 0.0]))
        exitos.append(True)
        prev_x = np.zeros(N)
        prev_lam = 0.0
        continue

    found = False
    seeds = generate_seeds(N, c_i, n_seeds=6)
    attempts = [prev_x] + seeds

    for x0 in attempts:
        z0 = np.hstack([x0, prev_lam])

        sol = root(
            system_constrained,
            z0,
            args=(a, b, c_i),
            method="hybr",
            tol=1e-10
        )

        if sol.success:
            x_sol = sol.x[:-1]
            lam_sol = sol.x[-1]

            
            if abs(np.dot(x_sol, x_sol) - c_i**2) < 1e-6:
                soluciones.append(sol.x)
                exitos.append(True)
                prev_x = x_sol
                prev_lam = lam_sol
                found = True
                break

    if not found:
        soluciones.append(np.full(N + 1, np.nan))
        exitos.append(False)
soluciones = np.array(soluciones)
exitos = np.array(exitos)

print("Total fallas:", np.sum(~exitos))
print("Ultimo c exitoso:", cs[exitos][-1] if np.any(exitos) else None)
print("\nResultados por valor de c:")
for c_i, ok in zip(cs, exitos):
    estado = "Éxito" if ok else "Falla"
    print(f"c = {c_i:4.1f}  ->  {estado}")


