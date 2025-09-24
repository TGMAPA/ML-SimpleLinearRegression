'''
Miguel Ángel Pérez Ávila 

Estructura para el control de procesos Machine Learning
'''

# Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import chardet

# Estructura para ejecutar procesos de Machine Learning 
class MLDev:
    # Metodo constructor de la clase para declarar los atributos de las instancias y recibir el path del archivo de datos
    def __init__(self):
        self.X = None
        self.Y = None
        self.Costos = None

    # Método para obtener el encoding del archivo de texto
    def getFileEncoding(self, path):
        # Detectar codificación
        with open(path, "rb") as f:
            raw_data = f.read(10000)
        encoding = chardet.detect(raw_data)["encoding"]

        return encoding

    # Método para realizar la lectura del archivo y llenar los arreglos de X y Y
    # - Se agrega automaticamente la columna de X0 con valores "unos" al vector X
    def readFile(self, path, chr2split):
        # Abrir archivo para lectura
        file = open(path, "r", encoding=self.getFileEncoding(path)) 
        self.X = []
        self.Y = []

        firstLine = True
        for line in file.readlines():
            if not firstLine:
                # Leer linea por linea y llenar la matriz de X y el arreglo de Y evitando el encabezado del dataset y asumiendo
                # que la columna de Y es la ultima en el dataset y Agregando la columna de 0 para X0 o bias

                line = line.split(chr2split)

                aux_xArray = [1.0] # Arreglo o temporal correspondiente a cada muestra del dataset (renglon)
                for i in range(len(line)):
                    if i == len(line)-1: # Ultimo elemento = Valor de y
                        self.Y.append( float(line[i]) )
                    else:
                        # Cualquier otro elemento = Valor de Xi
                        aux_xArray.append( float(line[i]))

                # Agregar arreglo temporal a la matriz X en construcción
                self.X.append(aux_xArray) 
                
            else:
                firstLine = False

        # Cast de list() a np.array
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        self.Y = self.Y.reshape(self.Y.shape[0], 1) # Redimensionar para corregir (m, ) a (m, 1)

    # Método para recalcular Pesos a partir de una Xi y un error=!=0 acierto paso definido
    def recalcW(self, W, X_train, Xi, step, error):
        for i in range(len(W)):
            W[i] = W[i] + (step * error * X_train[Xi][i])

    # Metodo para dividir el dataset en conjunto de prueba y entrenamiento para X y Y
    def dataset_division(self, train_percentage):

        X_train = self.X[:int(train_percentage*len(self.X))]
        X_test = self.X[int(train_percentage*len(self.X)):]

        Y_train = self.Y[:int(train_percentage*len(self.Y))]
        Y_test = self.Y[int(train_percentage*len(self.Y)):]

        return X_train, X_test, Y_train, Y_test
    
    # Método para entrenar y obtener los pesos ideales para un dataset 
    def simplePerceptron(self, step, train_percentage, n_epochs, f, W = np.array([]), RandW = True, graph = False):
        # Dividir dataset en arreglos de entrenamiento y de prueba
        X_train, X_test, Y_train, Y_test = self.dataset_division(train_percentage)

        if(RandW): # Calcular pesos aleatoriamente
            # Crear arreglo de pesos de dimensiones (N_VariablesXi, 1)
            W = np.random.uniform(low=0, high=1, size=(X_train.shape[1], 1))
        
        print("W Inicial:  \n", W)

        errors = np.array([]) # Arreglo para acumular errores por epoca
        epochs = np.arange(n_epochs) # Arreglo secuencial de 0-n_epochs

        # Iteración por epocas
        for i in range(n_epochs):
            # Aplicar multiplicación de matrices en vez de la sumatoria
            WX = np.dot(X_train, W)

            n_sample = 0 # Contador del número de muestra actual (numero de renglon Xi)
            for wx in WX: # Aplicar funcion de activación y calcular el error para cada muestra
                error = Y_train[n_sample]-f(wx)
                
                if  error == 0: # Error=0
                    pass
                else:
                    # Error != 0
                    # Recalcular Pesos
                    self.recalcW( W, X_train, n_sample, step, error)
                    break
                n_sample+=1

            # Acumular error obtenido en esta epoca, ya sea error!=0 o error==0
            errors = np.append(errors, error)     


        print("\nW Final: \n", W)

        W0 = W[0]
        W = W[1:]

        # CCrear funcion vecotrizada
        f = np.vectorize(f)


        # Predicción de datos de prueba
        X_test = np.delete(X_test, 0 ,axis= 1)
        Y_test_predicted = (np.dot(X_test, W)) + W0
        Y_test_predicted = f(Y_test_predicted)
        Y_test_predicted = Y_test_predicted.flatten()

        print("\n Predicción de datos de prueba")
        i = 0
        for y_predicted, y_desired in zip(Y_test_predicted, Y_test):
            print(i, " Desired: ", y_desired, "  | Predicted: ", y_predicted, "  | IsCorrect: ", y_desired==y_predicted)
            i+=1


        # Predicción de datos de entrenamiento para corroborar entrenamiento
        Y_train_predicted =  (np.dot(np.delete(X_train, 0 ,axis= 1), W)) + W0
        Y_train_predicted = f(Y_train_predicted)
        Y_train_predicted = Y_train_predicted.flatten()
        print("\n Predicción de datos de Entrenamiento")
        i = 0
        for y_predicted, y_desired in zip(Y_train_predicted, Y_train):
            print(i, " Desired: ", y_desired, "  | Predicted: ", y_predicted, "  | IsCorrect: ", y_desired==y_predicted)
            i+=1

        # Graficar Error
        if graph:
            self.graph(epochs, errors, "Epochs", "Error", "Error vs epochs")

        print("\nFinish")

    # Definir vector preConstruido para X - Considerando la existencia de columna bias x0 con unos.
    def setXVector(self, X):
        self.X = X

    # Definir veector preConstruido  para Y
    def setYVector(self, Y):
        self.Y = Y

    # Método para obtener el vector de Theta/Pesos Optimos mediante Gradiente Descendiente
    def gradDesc(self, n_epochs, alpha, thetas ):
        # Mostrar dimensiones de thetas y X
        print("\nThetas Shape: ", thetas.shape)
        print("X Shape     : ", self.X.shape)

        # Definir m como el tamaño de muestras que hay
        m = self.X.shape[0]

        # Arreglo de de almacenamiento de costos
        self.Costos = []
        
        # Redimensionar Y de (m, ) a (m,1)
        self.Y = self.Y.reshape(m, 1)
        print("Y Shape     : ", self.Y.shape)
        
        # Iteración por numero de epocas
        for i_epoch in range(n_epochs):
            # Calcular la "y-gorrito" como y_gorrito = X*Theta de forma vectorial
            y_hypothesis = np.dot(self.X, thetas)

            # Calcular el error 
            error = y_hypothesis - self.Y

            # Calculo de la gradiente unicuamente como  gradiente = (1/m)*error*X
            grad = (1/m) * np.dot(np.transpose(self.X), error)
            
            # Calculo final de thetas como  thetas = thetas - alpha * gradiente
            thetas = thetas - alpha*grad

            # Calculo del costo con las thetas actuales
            costo = self.calcCost(self.X, self.Y, thetas)
            self.Costos.append(costo)

        # Cast de list() a np.array
        self.Costos = np.array(self.Costos)
        
        return thetas

    # Función que calcula el costo para un vector de X. Y y thetas dadas
    def calcCost(self, x, y, thetas):
        # (97,2)*(2,1) = (97,1)
        f_hypothesis = (np.dot(x, thetas))  
        
        # ((97, 1) - (97, 1))**2 = (97, 1)
        toSum = ( f_hypothesis - y )**2   

        # Finalmente J(thetas) = (1/2m)(sumatoria)
        costo = ((1/(2*len(y)))*np.sum(toSum)) 

        return costo

    # Método para graficar dos variables
    def graph(self, x = None, y = None, xlabel="", ylabel="", title= ""):   
        if x == None:
            plt.plot(y)
        else:
            plt.plot(x, y)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()

    # Método para graficar 
    def scatterPlotAndLine(self, x, y, thetas):
        # ELiminar colmna del bias
        x = np.delete(x, 0 ,axis= 1) 

        # Dibujar dispersión
        plt.scatter(x, y, color="blue", label="Datos reales") 

        # Vector de valores de X 
        x = np.linspace(min(x), max(x), 100)  

        # Vector de valores resultantes de la ecuación y = theta0 + theta1 * x
        y = thetas[0] + thetas[1] * x    

        # Graficación
        plt.plot(x, y, color="red")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.title("Gráfico: Dispersión de Datos y Función Ajustada")
        plt.show()

