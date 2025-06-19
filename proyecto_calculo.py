import numpy as np
import matplotlib.pyplot as plt

# Constantes utilizadas
G  = 6.67430e-11            # Constante de gravitación universal [m^3 kg^-1 s^-2]
M  = 5.972e24               # Masa de la Tierra [kg]
GM = G * M                  # Producto útil [m^3 s^-2]
R_T = 6.371e6               # Radio medio de la Tierra [m]
h_everest = 8849.0          # Altura de monte Everest [m]

def f(r: float) -> float:
    """Aceleración gravitatoria en r (magnitud negativa: radial hacia el centro)."""
    return -GM / r**2

# 1. Taylor de orden 1 en r0 = R_T
f_RT      = f(R_T)
fprime_RT =  2 * GM / R_T**3

print("PUNTO 1 — Taylor orden 1 en r0 = R_T")
print(f"  f(R_T)      = {f_RT: .6e}  m/s²")
print(f"  f'(R_T)     = {fprime_RT: .6e}  m/s²·m⁻¹")
print("  P1(r) = f(R_T) + f'(R_T)*(r - R_T)")
print()

def taylor_p1(r: np.ndarray, r0: float = R_T) -> np.ndarray:
    return f(r0) + (2 * GM / r0**3) * (r - r0)

# 2. Variación a la altura del Everest (h = 8849 m)
r_plus_everest = R_T + h_everest
f_exact_everest = f(r_plus_everest)
relative_change = (f_exact_everest - f_RT) / f_RT

print("PUNTO 2 — Altura del Everest")
print(f"  f(R_T + h)        = {f_exact_everest: .6e}  m/s²")
print(f"  Diferencia rel.   = {relative_change*100: .4f}  %")
print("  g se reduce solo ~0.28 %, justificar g≈cte. es razonable.\n")

# 3. Taylor de orden 2 y comparación
fsecond_RT = -6 * GM / R_T**4

def taylor_p2(r: np.ndarray, r0: float = R_T) -> np.ndarray:
    h = r - r0
    return (f(r0) +
            (2 * GM / r0**3) * h +
            0.5 * fsecond_RT * h**2)

p2_everest = taylor_p2(r_plus_everest)
error_p2   = (p2_everest - f_exact_everest) / f_exact_everest

print("PUNTO 3 — ¿Sirve Taylor de orden 2?")
print(f"  P2(R_T + h)       = {p2_everest: .6e}  m/s²")
print(f"  Error relativo    = {error_p2*100: .4e}  %   (≈1x10⁻⁴ %)\n")

# 4. Gráfico cerca de R_T con P1 y P2
delta_graph = 2.0e5                                # ±200 km
h_vals = np.linspace(-delta_graph, delta_graph, 1000)
r_vals = R_T + h_vals

plt.figure(figsize=(8, 5))
plt.plot(h_vals/1000, f(r_vals), label="f(r) exacta")
plt.plot(h_vals/1000, taylor_p1(r_vals), '--', label="P1 (orden 1)")
plt.plot(h_vals/1000, taylor_p2(r_vals), ':', label="P2 (orden 2)")
plt.xlabel("h = r - R_T (km)")
plt.ylabel("a (m/s²)")
plt.title("Comparación cerca de R_T")
plt.legend()
plt.grid(True)
plt.show()

# 5. Altura para que g sea 1 % menor
target_factor = 0.99
r_target = R_T / np.sqrt(target_factor)
h_target = r_target - R_T

print("PUNTO 5 — Altura para que g sea 1 % menor")
print(f"  h ≈ {h_target/1000: .2f} km (≈ {h_target: .2f} m)\n")

# 6. Evaluación a r = 0.01 m y 0.02 m
r0_small   = 0.01           # 1 cm
r_double   = 0.02           # 2 cm
f_r0       = f(r0_small)
f_rdouble  = f(r_double)
relative_small = (f_rdouble - f_r0) / f_r0

# Taylor en r0_small
fprime_r0  =  2 * GM / r0_small**3
fsecond_r0 = -6 * GM / r0_small**4
fthird_r0  = 24 * GM / r0_small**5

def taylor_p2_small(r: np.ndarray) -> np.ndarray:
    h = r - r0_small
    return f_r0 + fprime_r0 * h + 0.5 * fsecond_r0 * h**2

def taylor_p3_small(r: np.ndarray) -> np.ndarray:
    h = r - r0_small
    return (f_r0 +
            fprime_r0 * h +
            0.5 * fsecond_r0 * h**2 +
            (1/6) * fthird_r0 * h**3)

p2_double = taylor_p2_small(r_double)
p3_double = taylor_p3_small(r_double)

err_p2 = (p2_double - f_rdouble) / f_rdouble
err_p3 = (p3_double - f_rdouble) / f_rdouble

print("PUNTO 6 — Radios 0.01 m y 0.02 m")
print(f"  Diferencia real        = {relative_small*100: .0f} % (-75 %)")
print(f"  P2(0.02 m) error rel.    = {err_p2*100: .0f} %")
print(f"  P3(0.02 m) error rel.    = {err_p3*100: .0f} % (cambia de signo)\n")

# 7. Gráfico cerca de r0 = 0.01 m con P2 y P3
r_plot = np.linspace(0.005, 0.025, 1000)   # 0.5 cm – 2.5 cm
plt.figure(figsize=(8, 5))
plt.plot(r_plot*100, f(r_plot), label="f(r) exacta")
plt.plot(r_plot*100, taylor_p2_small(r_plot), '--', label="P2 (orden 2)")
plt.plot(r_plot*100, taylor_p3_small(r_plot), ':', label="P3 (orden 3)")
plt.xlabel("r (m)")
plt.ylabel("a (m/s²)")
plt.title("Comparación cerca de r₀ = 0.01 m")
plt.legend()
plt.grid(True)
plt.show()
