import numpy as np
import matplotlib.pyplot as plt

# ========================= parámetros comunes ========================= #
g = 9.81  # gravedad (m/s²)

gammas = {               # γ (s⁻¹) para la Parte 1
    "Hormiga": 6.54,
    "Persona": 0.235,
    "Auto"   : 0.0105,
}
gamma_p2 = gammas["Persona"]   # usamos el caso 'Persona' en la Parte 2

# ========================= funciones de movimiento ==================== #
def y_exact(t: np.ndarray, gamma: float, y0: float, v0: float) -> np.ndarray:
    """Solución exacta con rozamiento lineal."""
    inv_gam = 1 / gamma
    coef = (v0 + g * inv_gam) * inv_gam
    return y0 - g * inv_gam * t - coef * (np.exp(-gamma * t) - 1)

def y_taylor(t: np.ndarray, gamma: float, y0: float, v0: float) -> np.ndarray:
    """Polinomio de Taylor de 2.º orden alrededor de t=0."""
    a0 = gamma * v0 + g
    return y0 + v0 * t - 0.5 * a0 * t**2

def y_libre(t: np.ndarray, y0: float, v0: float) -> np.ndarray:
    """Caída libre sin rozamiento."""
    return y0 + v0 * t - 0.5 * g * t**2

# PARTE 1                                
def parte1():
    """Reproduce los gráficos y la tabla de la Parte 1."""
    y0, v0 = 0.0, 0.0                       # condiciones iniciales fijas
    t = np.linspace(0, 2, 500)

    fig, axes = plt.subplots(len(gammas), 1, figsize=(8, 10), sharex=True)

    for ax, (nombre, gamma) in zip(axes, gammas.items()):
        ax.plot(t, y_exact(t, gamma, y0, v0), label='Exacta (con aire)')
        ax.plot(t, y_taylor(t, gamma, y0, v0), '--', label='Taylor 2º')
        ax.plot(t, y_libre(t, y0, v0), ':', label='Sin aire')
        ax.set_title(f"γ = {gamma:.4g} s⁻¹  ({nombre})")
        ax.set_ylabel("y(t) [m]")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Tiempo [s]")
    fig.tight_layout()
    plt.show()

    # -------- tabla de aceleraciones -------- # 
    # Se estudian los casos cuando vo = 0, vo < 0 y vo > 0
    print("\nCoeficiente cuadrático   (y''(0) = término de t²/2)\n"
          "-----------------------------------------------------")
    print("Caso       |  γ (s⁻¹) | Sin aire  |  Con aire")
    for nombre, gamma in gammas.items():
        a_free = -g
        a_air  = -g - gamma * v0
        print(f"{nombre:<10} | {gamma:8.4g} | {a_free:9.2f} | {a_air:9.2f}")
        # Nota: a_air = -g - γ·v0, que es el término de t²/2 en la Taylor 2º orden
    """
    NOTAS:
    Con γ pequeño:
        • El término γ·v0·t² es de orden menor ⇒ la curva con aire arranca casi igual
            que la de caída libre (y''(0) → -g cuando γ→0).
    Dependencia con el signo de v0:
        • v0  = 0  ⇒ y''(0) = -g; la resistencia no cambia la aceleración inicial.
        • v0 < 0   ⇒ γ·v0 < 0 ⇒ y''(0) > -g (menor módulo); el aire *reduce* la
            aceleración de caída.
        • v0 > 0   ⇒ γ·v0 > 0 ⇒ y''(0) < -g (mayor módulo en valor absoluto);
            el aire *aumenta* la desaceleración que frena la subida.
    """
        
# PARTE 2
def parte2():
    """Gráfica y compara los tres modelos para seis casos (y0, v0)."""
    cases = [
        (1,    0),
        (1000, 0),
        (1,   -100),
        (1000,-100),
        (0,    0.1),
        (0,   100),
    ]

    t = np.linspace(0, 20, 800)             # tiempo suficiente para ver diferencias
    rows, cols = 3, 2
    fig, axes = plt.subplots(rows, cols, figsize=(10, 12), sharex=True)

    for ax, (y0, v0) in zip(axes.ravel(), cases):
        ax.plot(t, y_exact(t, gamma_p2, y0, v0),       label='Exacta (con aire)')
        ax.plot(t, y_taylor(t, gamma_p2, y0, v0), '--', label='Taylor 2º')
        ax.plot(t, y_libre(t, y0, v0),            ':', label='Sin aire')
        ax.set_title(f"$y_0$={y0} m, $v_0$={v0} m/s")
        ax.set_ylabel("y(t) [m]")
        ax.grid(True)
        ax.legend(fontsize=8)

    for ax in axes[-1]:
        ax.set_xlabel("Tiempo [s]")

    fig.suptitle(
        f"Comparación de modelos (γ = {gamma_p2:.3f} s⁻¹) — Parte 2",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # -------- tabla de aceleraciones -------- #
    def a_inicial(func, *args, dt=1e-6):
        """Segunda derivada numérica en t = 0 mediante diferencia central."""
        return (func(dt, *args) - 2*func(0.0, *args) + func(-dt, *args)) / dt**2

    print("\nAceleración inicial y''(0)")
    print("-------------------------------------------")
    print(" y0 (m) |  v0 (m/s) |  Sin aire |  Con aire  | Taylor 2º orden")
    for y0, v0 in cases:
        a_free = a_inicial(lambda t, y0, v0: y_libre(t, y0, v0), y0, v0)
        a_air  = a_inicial(lambda t, y0, v0: y_exact(t, gamma_p2, y0, v0), y0, v0)
        a_taylor = a_inicial(lambda t, y0, v0: y_taylor(t, gamma_p2, y0, v0), y0, v0)
        print(f"{y0:7} | {v0:8} | {a_free:9.2f} | {a_air:9.2f} | {a_taylor:9.2f}")

    # -------- comentarios de las diferencias -------- #
    print("\n¿Cuándo difieren los modelos?")
    print("• Condiciones suaves (v0≈0, alturas bajas): las tres curvas casi se solapan "
          "durante varios segundos.\n"
          "• Velocidades iniciales grandes (±100 m/s): el término γ·v0 cambia la "
          "aceleración inicial; la curva exacta se separa rápido y la de Taylor "
          "sólo es válida cerca de t=0.\n"
          "• Alturas muy grandes (y0=1000 m): incluso con v0=0 el modelo exacto "
          "termina acercándose a la velocidad terminal; la caída libre sigue "
          "acelerando indefinidamente, por lo que la diferencia crece con el tiempo.\n"
          "• Si γ→0 (no mostrado) el efecto de rozamiento se atenúa y los tres "
          "modelos convergen a la parábola de caída libre.")

# PARTE 3
def v_exact(t: np.ndarray, gamma: float, v0: float) -> np.ndarray:
    """Velocidad exacta con rozamiento lineal (derivada de y_exact)."""
    return (v0 + g / gamma) * np.exp(-gamma * t) - g / gamma

def v_taylor(t: np.ndarray, gamma: float, v0: float) -> np.ndarray:
    """Velocidad del polinomio de Taylor de 2.º orden."""
    return v0 - (gamma * v0 + g) * t

def v_libre(t: np.ndarray, v0: float) -> np.ndarray:
    """Velocidad de la caída libre sin rozamiento."""
    return v0 - g * t

def parte3():
    """
    Muestra cómo la velocidad exacta se aproxima a v_T para cada γ.
    Se usa v0 = 0 m/s (caída desde el reposo) para resaltar la conver­gencia.
    """
    rows = len(gammas)
    fig, axes = plt.subplots(rows, 1, figsize=(9, 4 * rows), sharex=False)

    for ax, (nombre, gamma) in zip(axes, gammas.items()):
        v_T   = -g / gamma                     # velocidad terminal
        t_end = 6 / gamma                      # ~6 τ ⇒ e^{-6}≈0.002
        t = np.linspace(0, t_end, 600)

        # curvas
        ax.plot(t, v_exact(t, gamma, 0.0),       label='Exacta (con aire)')
        ax.plot(t, v_taylor(t, gamma, 0.0), '--', label='Taylor 2º')
        ax.plot(t, v_libre(t, 0.0),         ':',  label='Sin aire')
        ax.axhline(v_T, color='k', ls='-.', lw=1,
                   label=f"$v_T$ = {v_T:.1f} m/s")

        # formato del panel
        ax.set_title(f"γ = {gamma:.4g} s⁻¹  ({nombre})")
        ax.set_ylabel("v(t) [m/s]")
        ax.grid(True)
        ax.legend(fontsize=8, loc="best")

    axes[-1].set_xlabel("Tiempo [s]")
    fig.suptitle("Convergencia de la velocidad a $v_T$ para cada γ",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # ---- resumen numérico ----
    print("\nVelocidades terminales (caída desde el reposo)")
    print("----------------------------------------------")
    print("Caso      | γ (s⁻¹) |  v_T (m/s)")
    for nombre, gamma in gammas.items():
        print(f"{nombre:<9} | {gamma:7.4g} | {-g/gamma:10.2f}")
    print("\nObservaciones: \n")
    # 1° v0 != v:T
    print("• Con v0 != v_T: la velocidad converge exponencialmente a v_T \n"
          "                 con escala de tiiempo tc = 1/γ.\n"
          "                 Ej. para la persona, y'(t) está al 95\\% de v_t a los 13s\n"
          "• con v0 = v_T: el termino v0+g/γ = 0, entonces la part expoenencial\n"
          "                desaparece y toda la trayectoria va a velocidad constante v_T.\n"
          "• Con modelo libre y Taylor: ni y_libre ni y_taylor poseen v_T, la velocidad\n"
          "                             sigue creciendo indefinidamente (o decreciendo en caso de v0 < 0)\n"
          "                             con t(y'= vo -g*t) y no convergen a v_T.")
            
def parte4():
    """
    ¿Cuándo es válido ignorar el aire usando el desarrollo de Taylor?
    ---------------------------------------------------------------
    • Calcula el parámetro adimensional ε(t)≈|γ v0| t / |v0−½ g t|.
    • Muestra tablas para:
        1) Caída desde reposo  (v0 = 0)   → ε = γ t
        2) Lanzamiento rápido  (|v0| ≫ g t)
    • Da la regla práctica: γ t_mov ≲ 0.3 ⇒ error < 10 %.
    """
    def epsilon(t, gamma, v0):
        """Parámetro relativo entre Taylor con aire y sin aire."""
        denom = np.abs(v0 - 0.5 * g * t)
        # Evita división por cero o valores muy pequeños
        denom = np.where(denom < 1e-12, np.nan, denom)
        return np.abs(gamma * v0) * t / denom

    
    print("\n*** Parte 4 – Caída desde reposo (v0 = 0) ***")
    t_vals = np.array([0.5, 2.0, 6.0])           # s
    print("  t (s) |  γ   |  γ·t |  ¿ignorar aire? (γ·t ≤ 0.3)")
    for nombre, gamma in gammas.items():
        gt = gamma * t_vals
        decision = ["Sí" if x <= 0.3 else "No" for x in gt]
        for t, gt_i, dec in zip(t_vals, gt, decision):
            print(f"{t:6.2f} | {gamma:5.3f} | {gt_i:4.2f} | {dec}")

    print("\n*** Lanzamiento inicial rápido (|v0| = 50 m/s) ***")
    v0_fast = 50.0                              
    t_vals2 = np.array([0.1, 0.5, 1.0, 2.0])    
    print("  t (s)|   γ   |   ε(t)   |  ¿error <10 %? (ε≤0.1)")
    for nombre, gamma in gammas.items():
        eps = epsilon(t_vals2, gamma, v0_fast)
        decision = ["Sí" if e <= 0.1 else "No" for e in eps]
        for t, e_i, dec in zip(t_vals2, eps, decision):
            print(f"{t:6.2f} | {gamma:5.3f} | {e_i:8.3f} | {dec}")

    print(
        "\nRegla práctica (vía Taylor):  γ · t_mov  ≲  0.3\n"
        "Mientras se cumpla, el término cuadrático extra del modelo con aire\n"
        "no supera ~10 % del de caída libre, de modo que la resistencia lineal\n"
        "puede ignorarse con seguridad para estimaciones rápidas.\n"
    )
     
# ============================= ejecutar ============================== #
if __name__ == "__main__":
    parte1()   
    parte2()   
    parte3()
    parte4()
