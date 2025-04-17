# Detección de Tumores Cerebrales Mediante CNN

Este proyecto implementa una Red Neuronal Convolucional (CNN) para la detección de tumores cerebrales a partir de imágenes de Resonancia Magnética (RM). 
El dataset se extrajo de Kaggle utilizando el dataset [Brain tumors 256x256](https://www.kaggle.com/datasets/thomasdubail/brain-tumors-256x256) 

## Descripción del Proyecto

* **Clasificación Automática de Tumores:** El modelo clasifica imágenes de RM en cuatro categorías: glioma, meningioma, tumor pituitario y escáneres cerebrales normales.
* **Aumento de Datos:** Usamos técnicas de aumento de datos para mejorar la generalización del modelo, como reescalado de imágenes, rotación y volteo horizontal.

## Descripción del Dataset

El dataset utilizado en este proyecto es el "[Brain tumors 256x256](https://www.kaggle.com/datasets/thomasdubail/brain-tumors-256x256) 
". Contiene imágenes de RM categorizadas en las siguientes cuatro clases:

* Glioma Tumor
* Meningioma Tumor
* Pituitario Tumor
* Cerebro Normal

<img src="https://github.com/maugarciav/tc3002b-IA/blob/main/IMG/classes.png" height="500">

## Estructura del Proyecto

1.  **Carga de Datos:** Se utiliza la librería `kagglehub` para descargar el dataset directamente desde Kaggle.
2.  **Preprocesamiento de Datos:**
    * Reescalado de las imágenes para normalizar los valores de los píxeles.
    * Aumento de datos mediante rotación (hasta 18 grados) y volteo horizontal para incrementar la diversidad del conjunto de entrenamiento.
    * División del dataset en conjuntos de entrenamiento (80%) y validación (20%). 
3.  **Arquitectura del Modelo CNN (V1):**
    * El modelo es una red neuronal secuencial:
        * 3 capas Conv2D con función de activación ReLU.
        * 3 capas MaxPooling2D. 
        * 1 capa Flatten.
        * 1 capa Dropout (tasa de 0.4). 
        * 1 capa Densa con 256 nodos y función de activación ReLU. 
        * 1 capa Densa (capa de salida) con 4 nodos y función de activación Softmax. 
4.  **Entrenamiento del Modelo:**
    * Optimización: Optimizador Adam con una tasa de aprendizaje de 1e-4. 
    * Función de Pérdida: Sparse Categorical Crossentropy (para clasificación multiclase con etiquetas enteras)
    * Métricas: Precisión (Accuracy) para evaluar el rendimiento del modelo.
5.  **Evaluación del Modelo:**
    * Evaluación del modelo en el conjunto de validación para medir su capacidad de generalización.
    * Creamos una Matriz de confusión para analizar el rendimiento del modelo por clase.

## Resultados en Versión 1 (V1)

* Las gráficas de precisión y pérdida durante el entrenamiento muestran la evolución del rendimiento del modelo a lo largo de las 15 epochs
![](https://github.com/maugarciav/tc3002b-IA/blob/main/IMG/TrainValV1.png)
* Durante el entrenamiento, se realizó validación en cada época para comparar el rendimiento del modelo en datos no vistos. Podemos observar que existe un sobreajuste ya que la precisión en el conjunto de entrenamiento conitnua aumentando, mientras que en validación queda un poco estancada. Es decir, el modelo está memorizando los datos de entrenamiento en lugar de generalizar a nuevos ejemplos. De manera similar, la pérdida disminuye en el entrenamiento pero se estabiliza en la validación, confirmando este patrón.

    **Evaluación en el Conjunto de Validación:**
   
   * Precisión en el conjunto de validación: 76.15% 
   * Pérdida en el conjunto de validación: 0.9511
 

## Versión 2 (V2)

* En V1, se observó un sobreajuste del modelo, donde el rendimiento en el conjunto de entrenamiento era significativamente mejor que en el conjunto de validación.
* Para mitigar este problema, se exploraron estrategias de regularización basadas en el paper "[An Overview of Overfitting and its Solutions" de Xue Ying (2019)](https://iopscience.iop.org/article/10.1088/1742-6596/1168/2/022022/pdf)".
* De las técnicas descritas en el paper, se decidió implementar regularización L2 y ajustar la tasa de dropout. (La regularización L2 se implementó utilizando la función `kernel_regularizer` de Keras).
* Todos los demás aspectos de la arquitectura del modelo y el proceso de entrenamiento se mantuvieron sin cambios respecto a V1.

   ### TEST 1: V2 - Regularization (Strength=1e-4, Dropout=0.4)
   
   * Para esta primera iteración, se aplicó regularización L2 a las capas Conv2D y la capa Densa intermedia con un strength de 1e-4, manteniendo la tasa de dropout en 0.4. 
   
   * Gráficas de precisión y pérdida:
   
       ![](https://github.com/maugarciav/tc3002b-IA/blob/main/IMG/TrainValV2-Test1.png)
   
       * Evaluación en el Conjunto de Validación:
   
           * Precisión en el conjunto de validación: 79.28%
           * Pérdida en el conjunto de validación: 0.8031
   
      La implementación de la regularización L2 demostró una mejora en el rendimiento del modelo en comparación con V1. Hubo un incremento en la precisión del conjunto de validación (de 76.15% a 79.28%) y una disminución en la pérdida de validación         (de 0.9511 a 0.8031). Esto indica que la regularización ayudó a reducir el sobreajuste y mejorar la generalización del modelo. Sin embargo, las gráficas de entrenamiento y validación aún muestran cierto grado de sobreajuste.
   
   
   
   ### Test 2: V2 - Strong Regularization (Strength=1e-3, Dropout=0.5)
   
   * En esta segunda iteración de V2, se aplicó una regularización L2 más fuerte (strength de 1e-3) y una tasa de dropout aumentada (0.5) con el objetivo de reducir aún más el sobreajuste observado en las pruebas anteriores.
   
   * Gráficas de precisión y pérdida:
   
        ![](https://github.com/maugarciav/tc3002b-IA/blob/main/IMG/TrainValV2-Test2.png)
   
      * Evaluación en el Conjunto de Validación:
      
          * Precisión en el conjunto de validación: 81.09%
          * Pérdida en el conjunto de validación: 0.9722
   
      Los resultados de este test indican una mejora en la precisión del modelo en el conjunto de validación, obteniendo una mejor generalización. Además, las gráficas muestran una menor divergencia entre las curvas de entrenamiento y validación, lo        que confirma una reducción del sobreajuste. Sin embargo, la pérdida en el conjunto de validación *aumentó* a 0.9722 (en comparación con 0.8031 en Test 1). Es decir, el modelo está clasificando más imágenes correctamente, pero la "confianza"           promedio en las predicciones es menor. Esto refleja el efecto de la mayor regularización. En general, este modelo muestra una compensación entre precisión y confianza, con una mejor capacidad para generalizar a expensas de una mayor pérdida.
      
   
   
   ### Test 3: V2 - Moderate Regularization (Strength=5e-4, Dropout=0.45)
   
   * En esta tercera iteración de V2, se ajustó la regularización L2 a un strength de 5e-4 y la tasa de dropout a 0.45, buscando un punto intermedio entre las configuraciones de los Test 1 y Test 2.
   
   * Gráficas de precisión y pérdida:
   
       ![](https://github.com/maugarciav/tc3002b-IA/blob/main/IMG/TrainValV2-Test3.png)
   
   * Evaluación en el Conjunto de Validación:
   
       * Precisión en el conjunto de validación: 81.91%
       * Pérdida en el conjunto de validación: 0.8296
   
       En este Test el modelo mantiene la alta precisión de validación alcanzada en el Test 2 y, al mismo tiempo, logra reduce significativa en la pérdida de validación (0.8296). Esta configuración logra un mejor equilibrio entre la capacidad del            modelo para generalizar a datos no vistos y la confianza en sus predicciones. Las gráficas también confirman una menor divergencia general entre las curvas de entrenamiento y validación, indicando una reducción del sobreajuste, sin                    embargo, en las curvas de pérdida, especialmente la de validación, hay fluctuaciones o "saltos" pronunciados, esto podría indicar cierta inestabilidad en el proceso de aprendizaje, lo cual es algo a tomar en cuenta. En fin a comparación de los        tests anteriores, el Test 3 nos da los mejores resultados entre precisión y pérdida.



##   Versión 3 (V3)

* En V3, se implementó la arquitectura y las técnicas de regularización descritas en el paper "[Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)" de Srivastava, Hinton, Krizhevsky, Sutskever, y Salakhutdinov (2014).Este trabajo introduce la técnica de "dropout" como método para reducir el overfitting en redes neuronales profundas. Buscamos tener menor fluctuación en las gráficas de precisón y pérdida lo cual indcaría una mejor estabilidad en el proceso de aprendisaje.
* Los cambios clave con respecto a V2 son:

    * **Dropout Placement:** Se aplicó Dropout después de cada capa MaxPooling2D en las capas convolucionales, además de las capas densas.
    * **Max-Norm Regularization:** Se añadió regularización Max-Norm a la capa Dense intermedia.

* La arquitectura es la siguiente:

    * Capas Convolucionales:
        * Conv2D (32 filtros, (3x3), ReLU)
        * MaxPooling2D ((2x2))
        * Dropout (0.2)
        * Conv2D (64 filtros, (3x3), ReLU)
        * MaxPooling2D ((2x2))
        * Dropout (0.2)
        * Conv2D (128 filtros, (3x3), ReLU)
        * MaxPooling2D ((2x2))
        * Dropout (0.2)
    * Capas Densas:
        * Flatten
        * Dropout (0.4)
        * Dense (256 neuronas, ReLU, MaxNorm constraint = 3)
        * Dropout (0.4)
        * Dense (num\_classes, Softmax)
* **Entrenamiento:**

    * Optimizador: Adam (learning rate = 1e-4)
    * Pérdida: Sparse Categorical Crossentropy
    * Métricas: Accuracy


   ### Test 1: V3
   
   * Gráficas de precisión y pérdida:
   
       ![](https://github.com/maugarciav/tc3002b-IA/blob/main/IMG/TrainValV3-Test1.png)
   
   * Evaluación en el Conjunto de Validación:
 
      * Precisión en el conjunto de validación: 78.45%
      * Pérdida en el conjunto de validación: 0.5771

   Los resultados de V3 muestran una mejora en la pérdida en comparación con V1 y V2, aunque la precisión es ligeramente inferior a la mejor precisión obtenida en V2. Esta disminución en la pérdida nos indica que el modelo V3 está realizando             predicciones con mayor confianza. Además, tenemos un aprendizaje mucho más controlado y estable durante el entrenamiento, que era justo lo que buscabamos. En general, V3 demuestra que la aplicación de las técnicas de Dropout y Max-Norm, tal como      se propone en el paper de Srivastava, tiene un impacto significativo en la regularización del modelo, ayudando a un aprendizaje más estable y mejorando la confianza en las predicciones.


## Autor

Mauricio Garcia Villanuvea

## Referencias

Ying, X. (2019). An Overview of Overfitting and its Solutions. IOP Conference Series: Journal of Physics: Conference Series, 1168(2), 022022. https://iopscience.iop.org/article/10.1088/1742-6596/1168/2/022022/pdf


## Licencia

[Licencia del proyecto](https://creativecommons.org/publicdomain/zero/1.0/)
