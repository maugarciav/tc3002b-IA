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

![](https://github.com/maugarciav/tc3002b-IA/blob/main/IMG/classes.png)

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
* De las técnicas descritas en el paper, se decidió implementar regularización L2 y ajustar la tasa de dropout.
* Todos los demás aspectos de la arquitectura del modelo y el proceso de entrenamiento se mantuvieron sin cambios respecto a V1.

### TEST 1 -> V2 - L2 Regularization (Strength=1e-4, Dropout=0.4)

* Para esta primera iteración, se aplicó regularización L2 a las capas Conv2D y la capa Densa intermedia con un strength de 1e-4, manteniendo la tasa de dropout en 0.4. (La regularización L2 se implementó utilizando la función `kernel_regularizer` de Keras).

* Gráficas de precisión y pérdida:

    ![](https://github.com/maugarciav/tc3002b-IA/blob/main/IMG/TrainValV2-Test1.png)

    * Evaluación en el Conjunto de Validación:

        * Precisión en el conjunto de validación: 79.28%
        * Pérdida en el conjunto de validación: 0.8031

La implementación de la regularización L2 demostró una mejora en el rendimiento del modelo en comparación con V1. Hubo un incremento en la precisión del conjunto de validación (de 76.15% a 79.28%) y una disminución en la pérdida de validación (de 0.9511 a 0.8031). Esto indica que la regularización ayudó a reducir el sobreajuste y mejorar la generalización del modelo. Sin embargo, las gráficas de entrenamiento y validación aún muestran cierto grado de sobreajuste.

##  Mejoras para V2
* **Arquitecturas de Modelo Avanzadas:** Probar con arquitecturas CNN más complejas o modelos pre-entrenados (como VGG16, ResNet o EfficientNet) para mejorar la precisión.
* **Aumento de Datos Extensivo:** Probar técnicas de aumento de datos más avanzadas para tener un modelo mas robusto.
* **Regularización Adicional:** Usar más técnicas de regularización, como normalización por lotes (Batch Normalization), para reducir el sobreajuste.


## Autor

Mauricio Garcia Villanuvea

## Licencia

[Licencia del proyecto](https://creativecommons.org/publicdomain/zero/1.0/)
