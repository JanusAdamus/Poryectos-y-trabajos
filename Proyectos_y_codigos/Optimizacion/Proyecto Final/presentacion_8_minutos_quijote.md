# Presentación del Proyecto Final

## Diapositiva 1. Título

**Embeddings de palabras con optimización restringida en un fragmento de Don Quijote**

- Integrantes: [Nombres]
- Materia: [Nombre]
- Profesor: [Nombre]

**Qué decir**

“En este proyecto usamos un fragmento de *Don Quijote de la Mancha* para construir embeddings de palabras. La idea fue comparar dos formas de resolver el mismo problema de optimización restringida: penalización cuadrática y barrera logarítmica.”

## Diapositiva 2. Problema

- Queríamos representar palabras como vectores en 2D.
- Tomamos palabras clave del fragmento.
- Si dos palabras aparecen en contextos parecidos, sus vectores deberían quedar cerca.

**Qué decir**

“Nos enfocamos en un problema pequeño para que el resultado fuera fácil de interpretar. En lugar de trabajar con toda la novela, usamos un fragmento corto y nos quedamos con catorce palabras importantes como Quijote, Sancho, Rocinante, Dulcinea, molinos y viento.”

## Diapositiva 3. Datos

- Texto base: fragmento de *Don Quijote*.
- Vocabulario final:
  `quijote`, `sancho`, `rocinante`, `dulcinea`, `molinos`, `viento`, `caballero`, `aventuras`, `mancha`, `gigantes`, `batalla`, `escudero`, `novela`, `juicio`
- Ventana de contexto: 2 palabras.

**Qué decir**

“Primero limpiamos el texto, quitamos palabras vacías y dejamos solo términos que sí aportaban al análisis. Luego formamos pares de co-ocurrencia con una ventana de contexto de tamaño dos.”

## Diapositiva 4. Modelo

- Función de pérdida tipo *skip-gram* con muestreo negativo.
- Restricción 1: cada embedding debe tener norma menor o igual a 1.
- Restricción 2: las dimensiones deben mantenerse lo más ortogonales posible.

**Qué decir**

“El modelo premia a las palabras que sí aparecen juntas y castiga pares aleatorios. Además, no dejamos que los vectores crezcan sin control y buscamos que las dos dimensiones no representen lo mismo.”

## Diapositiva 5. Métodos comparados

- **Penalización cuadrática**
  Suma un castigo si se violan las restricciones.
- **Barrera logarítmica**
  Empuja la solución a quedarse dentro de la región factible.

**Qué decir**

“Los dos métodos atacan el mismo problema, pero de forma distinta. La penalización corrige cuando ya hay violación. La barrera trata de evitar acercarse demasiado al borde desde el principio.”

## Diapositiva 6. Resultados numéricos

| Método | Data loss | Error ortogonalidad | Máx. norma² | Holgura mínima |
|---|---:|---:|---:|---:|
| Penalización | 1.3454 | 0.0138 | 0.2887 | 0.7113 |
| Barrera | 1.3473 | 0.0124 | 0.2288 | 0.7712 |

**Qué decir**

“Los dos métodos ajustaron casi igual los datos, porque la pérdida final fue muy parecida. La diferencia apareció en las restricciones: la barrera terminó con más holgura y menor norma máxima, así que dejó una solución más interior.”

## Diapositiva 7. Interpretación

- Vecinos cercanos con barrera:
  - `quijote -> dulcinea, molinos, rocinante`
  - `sancho -> escudero, viento, molinos`
  - `molinos -> quijote, dulcinea, viento`
- La estructura aprendida sí refleja parte del relato.

**Qué decir**

“Aunque el texto es corto, el embedding sí captó relaciones razonables. Por ejemplo, Sancho quedó cerca de escudero, y Quijote quedó cerca de Rocinante y Dulcinea. Eso muestra que el contexto sí organizó las palabras de manera coherente.”

## Diapositiva 8. Gráfica

Usar la imagen `embeddings_quijote.png`.

**Qué decir**

“Aquí se ve la posición final de las palabras en dos dimensiones. El círculo punteado marca la restricción de norma. Visualmente también se nota que ambos métodos funcionan, pero la barrera deja una distribución un poco más contenida.”

## Diapositiva 9. Conclusión

- El modelo mínimo sí funciona.
- La barrera y la penalización ajustan parecido.
- La barrera respetó mejor las restricciones.
- El caso de *Don Quijote* fue suficiente para mostrar el enfoque.

**Qué decir**

“Nuestra conclusión es que el experimento cumple con el objetivo del curso: formular un problema de optimización restringida, resolverlo con dos enfoques y comparar resultados. En este caso, la barrera dio una solución más limpia sin perder calidad en el ajuste.”

## Reparto del tiempo

1. Introducción: 1 minuto
2. Datos y modelo: 2 minutos
3. Métodos: 1.5 minutos
4. Resultados: 2 minutos
5. Conclusión: 1.5 minutos

## Material a mostrar

1. El archivo `quijote_minimo.py`
2. La tabla de resultados
3. La imagen `embeddings_quijote.png`
