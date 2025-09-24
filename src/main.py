'''
Miguel Ángel Pérez Ávila

Archivo main con un ejemplo de uso en la función main()

'''

# Importar Librerias y modulos
from MLDev import MLDev
from SimpLinealReg import gradienteDescendente, graficaDatos, calcularCosto
import numpy as np


# Función main para la ejecución de ejemplo
def main():

    # ====== Ejemplo de ejecución de las funciones implementadas ======

    mlObject = MLDev() # Instancia para el manejo de procesos de ML
    path = "../data/ex1data1.txt" 

    # Lectura de archivo mediante un caracter separador y llenado de vectores
    mlObject.readFile(path, "\t")  
    
    # Vector de thetas iniciales tamaño (n_variables, 1)
    thetas = np.zeros((mlObject.X.shape[1], 1))
    print("\nThetas Iniciales: \n", thetas)

    # Thetas óptimas resultantes del Entrenamiento en busca de thetas óptimas mediante gradiente descendente
    thetas = gradienteDescendente(mlObject.X, mlObject.Y, thetas, 0.01, 1500)
    print("\nThetas Final    : \n", thetas)

    print("\nPredicciones")
    print(np.dot(np.array([1, 3.5]), thetas))
    print(np.dot(np.array([1, 7]), thetas))

    # Graficación de disperción con función ajustada visible
    print("\nGraficando disperción y recta ajustada con valores theta: \n", thetas)
    graficaDatos(mlObject.X, mlObject.Y, thetas)

    # Ejemplo de cálculo de costo
    print("\nEjemplo de calculo de costo:")
    print(calcularCosto(mlObject.X, mlObject.Y, thetas))

# Ejecutar Main
main()