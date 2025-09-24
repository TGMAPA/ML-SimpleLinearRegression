# ML-SimpleLinearRegression: Regresión Lineal Simple con Gradiente Descendente

Este proyecto presenta la implementación manual de un modelo de **Regresión Lineal Simple** en Python, utilizando gradiente descendente para optimizar los parámetros del modelo (θ). El objetivo es permitir la predicción de valores continuos a partir de una sola característica, proporcionando comprensión teórica y aplicación práctica del algoritmo.

El proyecto incluye carga y preparación de datos, entrenamiento del modelo, cálculo de la función de costo, visualización de resultados y predicciones para nuevos valores de entrada.

---

## Objetivo
- Implementar un modelo de regresión lineal simple de forma manual.  
- Optimizar los parámetros utilizando gradiente descendente.  
- Visualizar la dispersión de los datos y la recta ajustada.  
- Realizar predicciones para nuevas muestras y evaluar el desempeño del modelo mediante la función de costo.  

---

## Metodología
El proyecto utiliza un enfoque orientado a objetos mediante la clase `MLDev`, que proporciona la estructura y funciones necesarias para el manejo completo del modelo:

### Funcionalidades principales de `MLDev`
1. **Carga y preparación de datos**
   - `getFileEncoding(path)`: Detecta la codificación del archivo de datos.  
   - `readFile(path, separador)`: Lee el dataset y construye las matrices X (incluye columna bias) y Y.  
   - `setXVector(X)`: Define manualmente el vector/matriz de características.  
   - `setYVector(Y)`: Define manualmente el vector de etiquetas.  

2. **Entrenamiento del modelo**
   - `gradDesc(n_epochs, alpha, thetas)`: Optimiza los parámetros mediante gradiente descendente.  
   - `calcCost(X, Y, thetas)`: Calcula la función de costo J(θ).  

3. **Visualización**
   - `graph(x, y, xlabel, ylabel, title)`: Grafica la evolución del costo a través de las épocas.  
   - `scatterPlotAndLine(X, Y, thetas)`: Grafica la dispersión de los datos y la recta ajustada.  

### Funciones adicionales implementadas
- `graficaDatos(x, y, theta)`: Grafica los datos y la recta ajustada para un conjunto de X y Y dado.  
- `gradienteDescendente(x, y, thetas, alpha, iteraciones)`: Optimiza el vector de θ mediante gradiente descendente.  
- `calcularCosto(x, y, thetas)`: Calcula la función de costo para un vector de X, Y y θ dado.

---

 ## Ejecución
1. Clonar este repositorio:  
   ```bash
   git clone https://github.com/TGMAPA/ML-SimpleLinearRegression.git
   cd ML-SimpleLinearRegression/src
   python main.py
