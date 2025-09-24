'''
Miguel Ángel Pérez Ávila

Archivo con la implementación de funciones utilizadas

'''

# Importar Librerias y modulos
from MLDev import MLDev


# ========== Funciones Implementadas ==========

# Función para graficar la disperción de datos (Dados los Vectores X y Y) y la función Recta ajustada
# Está función ASUME que el vector de X aún contiene la columna X0 del bias con "unos"  
# - Input:
#       - x: Vector de X (Aún conteniendo la columna de X0)
#       - y: Vector de Y 
#       - theta: Vector de thetas óptimas para la recta
# - Return : Void
def graficaDatos(x, y, theta):
    mlObject = MLDev() # Instancia para el manejo de procesos de ML
    mlObject.scatterPlotAndLine(x,y,theta) # Ejecución del método implementado para la graficación

# Función para optimizar y conseguir el vector de thetas optimo que ajuste a los vecotres de datos dados a razon
# de un paso (alpha) definido y una n cantidad de iteraciones
# - Input:
#       - x: Vector de X (YA conteniendo la columna de X0 con "unos")
#       - y: Vector de Y 
#       - thetas: Vector de thetas iniciales para el entrenamiento
#       - aplha: Learning Rate o paso para el ajuste de pesos
#       - iteraciones: epocas a ejecutar
# - Return : 
#       - thetasGradiente: np.array de thetas óptimas
def gradienteDescendente(x, y, thetas, alpha, iteraciones):
    mlObject = MLDev() # Instancia para el manejo de procesos de ML
    mlObject.setXVector(x) # Método para definir manualmente un vector X considerando que YA contiene la columna para X0 con "unos"
    mlObject.setYVector(y) # Método para definir manualmente un vector Y de tamaño (mx1)
    
    # Thetas óptimas resultantes del Entrenamiento en busca de thetas óptimas mediante gradiente descendente
    thetasGradiente = mlObject.gradDesc(iteraciones, alpha, thetas)

    # Graficación de costos en epocas
    print("Mostrando gráfico de costo a través de las epocas...")
    mlObject.graph(None, mlObject.Costos, "Epocas", "Costo ( J(theta) )", "Gráfico: Costo vs Epocas")

    # Retornar vector de thetas óptimas calculadas
    return thetasGradiente

# Función para calcular el costo dado un Vector de X, Y y thetas
# - Input:
#       - x: Vector de X (Conteniendo la columna de X0)
#       - y: Vector de Y 
#       - theta: Vector de thetas a evaluar
# - Return : 
#       - Costo calculado como: J(thetas) = (1/2m)(sumatoria(( f_hypothesis - y )**2))
def calcularCosto(x, y, thetas):
    mlObject = MLDev() # Instancia para el manejo de procesos de ML
    return mlObject.calcCost(x, y, thetas)

# ========== Fin de Funciones ==========

