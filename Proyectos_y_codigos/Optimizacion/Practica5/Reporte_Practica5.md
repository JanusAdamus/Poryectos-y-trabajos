# Respuestas Práctica 5

**Pregunta 1.a:** Haz la minimización libre por los dos métodos mencionados. Para el método de Newton, demuestra que el Hessiano es positivo definido.

**Respuesta:**  
Para \(58x^2 + 85y^2 - 34xy\), el mínimo libre teórico es \([0,0]\).  
Descenso por gradiente llegó a \([-0., 0.]\) con error \(2.6988324267725293\times 10^{-13}\).  
Newton llegó a \([0.,0.]\) con error \(0.0\).  
El Hessiano es positivo definido porque sus autovalores son \([99.583413,\ 186.416587]\).

**Evidencia:**

```text
Cuadratica: Hessiano libre PD porque sus autovalores son [ 99.583413 186.416587]

Cuadratica
libre GD      [-0.  0.] error = 2.6988324267725293e-13
libre Newton  [0. 0.] error = 0.0
```

**Pregunta 1.b:** Haz la minimización restringida a la línea \(y=-x-1\). Numéricamente prueba si el Hessiano es positivo definido.

**Respuesta:**  
El mínimo sobre la recta es \([-0.576271,\ -0.423729]\).  
Numéricamente, el Hessiano restringido es positivo definido porque \(z^T H z = 354.0 > 0\).

**Evidencia:**

```text
Cuadratica en recta: z^T H z = 354.0
recta Newton  [-0.576271 -0.423729] error = 0.0
```

**Pregunta 1.c:** Haz la minimización restringida al interior del círculo \((x-1)^2+y^2=1\). Comprueba, en cada paso, si el Hessiano es positivo definido. De no ser así, modifícalo para que lo sea.

**Respuesta:**  
La solución reportada en el notebook para el método de barrera es \([0.001711,\ 0.000342]\) con error \(0.001744891832407594\).  
No fue necesario corregir el Hessiano: hubo \(0\) correcciones y el desplazamiento máximo fue \(0.0\).

**Evidencia:**

```text
disco barrera [0.001711 0.000342] error = 0.001744891832407594
correcciones Hessiano = 0 max shift = 0.0
```

![](evidencias_practica5/evidencia_1.png)

**Pregunta 2.a:** Repite el problema anterior para la banana de Rosenbrock en 2 dimensiones, haciendo la minimización libre por descenso de gradiente y Newton.

**Respuesta:**  
Para \((1-x)^2 + 100(y-x^2)^2\), el mínimo libre teórico es \([1,1]\).  
Descenso por gradiente llegó a \([1.001621,\ 1.003245]\) con error \(0.0036273634492732104\).  
Newton llegó a \([1.,1.]\) con error \(0.0\).

**Evidencia:**

```text
Rosenbrock
libre GD      [1.001621 1.003245] error = 0.0036273634492732104
libre Newton  [1. 1.] error = 0.0
```

**Pregunta 2.b:** Repite la minimización restringida a la línea \(y=-x-1\) para la función de Rosenbrock. Puedes estudiar sus valores propios ya sea analíticamente o numéricamente.

**Respuesta:**  
El mínimo sobre la recta es \([-0.490068,\ -0.509932]\).  
El criterio analítico usado en el notebook fue \(d^2/dt^2 = 1200t^2 + 1200t + 602\), con discriminante \(-1449600\), por lo que es positiva para todo \(t\).

**Evidencia:**

```text
Rosenbrock en recta: d2/dt2 = 1200 t^2 + 1200 t + 602
discriminante = -1449600 < 0, entonces es positiva para todo t
recta Newton  [-0.490068 -0.509932] error = 1.0605390551611693e-15
```

**Pregunta 2.c:** Repite la minimización restringida al interior del círculo para la función de Rosenbrock y comprueba si el Hessiano requiere correcciones.

**Respuesta:**  
La solución reportada por el método de barrera es \([0.987507,\ 0.975107]\) con error \(0.027851999061028535\).  
Sí hubo correcciones al Hessiano: \(2\) correcciones y desplazamiento máximo \(135.1019725339529\).

**Evidencia:**

```text
disco barrera [0.987507 0.975107] error = 0.027851999061028535
correcciones Hessiano = 2 max shift = 135.1019725339529
```

![](evidencias_practica5/evidencia_2.png)
