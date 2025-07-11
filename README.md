﻿# Proyecto de Cálculo: Polinomios de Taylor Aplicados

Este proyecto muestra cómo aplicar desarrollos de Taylor de primer, segundo y tercer orden para resolver y analizar problemas de física clásica, en particular la gravitación universal y el movimiento con rozamiento en caída libre.

Incluye dos ejercicios desarrollados en Python:

1. **Modelo de Gravitación Universal** (`ej1_modelo_gravitacional.py`)
2. **Caída Libre con Rozamiento Lineal** (`ej2_caida_libre.py`)


## 1. Requisitos

Este proyecto requiere:

- Python 3.8 o superior
- Bibliotecas externas:

```bash
pip install numpy matplotlib
```


## 2. Estructura del Proyecto

```
📁 Proyecto_Taylor
├── ej1_modelo_gravitacional.py   # Ejercicio 1
├── ej2_caida_libre.py            # Ejercicio 2
├── README.md                     # Este archivo
├── main.pdf                      # Informe con desarrollo completo
```

## 3. Ejecución

Desde línea de comandos (cmd o terminal en VS Code):

```bash
python .\ej1_modelo_gravitacional.py
python .\ej2_caida_libre.py
```

## 4. Ejercicio 1: Modelo de Gravitación Universal

Se analiza la función:

$$
f(r)= -\dfrac{GM}{r^{2}}
$$

Y se aproxima usando polinomios de Taylor centrados en el radio terrestre. Se grafican los valores exactos y aproximados, se analizan errores relativos, y se explora el comportamiento de la gravedad a diferentes alturas (Everest, radios pequeños).

**El programa:**

- Imprime en consola resultados numéricos.
- Muestra dos gráficos:
  - Gravedad vs. altura sobre la Tierra.
  - Gravedad para radios muy pequeños.

## 5. Ejercicio 2: Caída Libre con Rozamiento Lineal

Se modela el movimiento vertical bajo gravedad y rozamiento:

$$
y(t) = y_0 - \dfrac{g}{\gamma}t - \dfrac{1}{\gamma}\left(v_0 + \dfrac{g}{\gamma}\right)\left(e^{-\gamma t} - 1\right)
$$

Se compara con:

- Movimiento sin aire.
- Aproximación de Taylor de orden 2.
- Se calculan velocidades terminales y errores en distintos escenarios.

**Incluye:**

- Evaluación con diferentes valores iniciales (`y0`, `v0`)
- Estimación de cuándo se puede despreciar el aire
- Comparación gráfica entre los modelos

## 6. Cómo interpretar los resultados

- Se imprime paso a paso cada parte del cálculo.
- Se explicita el uso de Taylor para evaluar errores.
- Las gráficas comparan soluciones exactas con aproximaciones.

---

**Desarrollado por:**  
Luis Balduini, Leandro Casaretto y Juan Pérez  
**Curso:** Cálculo Aplicado, Universidad Católica del Uruguay
