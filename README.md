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

## Conceptos
* **Recall (Sensibilidad):** Esta métrica es de suma importancia en el contexto de la detección de tumores, ya que mide la capacidad del modelo para identificar correctamente todos los casos positivos reales de cada tipo de tumor y de escáneres normales. Un **Recall alto (cercano a 1)** indica que el modelo es efectivo minimizando los **falsos negativos** (es decir, la situación en la que un tumor está presente pero el modelo no lo detecta). El Recall varía **entre 0 y 1**.

* **F1-Score:** El F1-Score proporciona una visión equilibrada del rendimiento del modelo, considerando tanto la **precisión** (¿cuántas de las predicciones positivas realizadas por el modelo fueron realmente correctas?) como el **recall**. Un **F1-Score alto (cercano a 1)** significa que el modelo tiene un buen equilibrio entre precisión y sensibilidad. Esta métrica es especialmente útil cuando las clases podrían estar ligeramente desbalanceadas. El F1-Score también varía **entre 0 y 1**.

* **Matriz de Confusión:** La matriz de confusión es una herramienta visual para entender dónde el modelo está cometiendo errores. Muestra el número de instancias que fueron clasificadas correctamente e incorrectamente para cada par de clases verdadera y predicha, revelando patrones de confusión entre ellas.

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

* **Técnicas de Regularización Implementadas:**
    * **Regularización L2 (Weight Decay):** De las técnicas descritas en el paper de Ying (2019), se decidió implementar la regularización L2. Esta técnica penaliza la complejidad del modelo añadiendo un término a la función de pérdida que es proporcional a la suma de los cuadrados de los pesos de la red. La idea es que los modelos con pesos más pequeños tienden a ser más simples y generalizan mejor. Al minimizar la función de pérdida modificada, el optimizador se ve incentivado a mantener los pesos pequeños, evitando que algunos pesos individuales crezcan excesivamente y dominen las predicciones. Esto reduce la dependencia del modelo de características específicas (y potencialmente ruidosas) de los datos de entrenamiento. Esto se implementó usando `kernel_regularizer=regularizers.l2(strength)` en las capas Conv2D y Dense, donde `strength`.
    * **Dropout:** Se ajustó la tasa de dropout aplicada después de la capa Flatten para complementar la regularización L2.

* **Arquitectura del Modelo V2:** La arquitectura base se mantuvo similar a V1, pero incorporando regularización L2 en capas convolucionales y densas, y ajustando la tasa de Dropout:
    * **Capas Convolucionales:**
        * `Conv2D` (32 filtros, (3x3), ReLU, **Regularizador L2**)
        * `MaxPooling2D` ((2x2))
        * `Conv2D` (64 filtros, (3x3), ReLU, **Regularizador L2**)
        * `MaxPooling2D` ((2x2))
        * `Conv2D` (128 filtros, (3x3), ReLU, **Regularizador L2**)
        * `MaxPooling2D` ((2x2))
    * **Capas Densas:**
        * `Flatten`
        * `Dropout`
        * `Dense` (256 neuronas, ReLU, **Regularizador L2**)
        * `Dense` (num\_classes, Softmax)  
   * **Entrenamiento:**
       * Optimizador: Adam (learning rate = 1e-4)
       * Pérdida: Sparse Categorical Crossentropy
       * Métricas: Accuracy

   ### TEST 1: V2 - Regularization (Strength=1e-4, Dropout=0.4)
   
   * En la primera iteración, se aplicó regularización L2 a las capas Conv2D y la capa Densa intermedia con un strength de 1e-4, manteniendo la tasa de dropout en 0.4. 
   
   * Gráficas de precisión y pérdida:
   
       ![](https://github.com/maugarciav/tc3002b-IA/blob/main/IMG/TrainValV2-Test1.png)
   
       * Evaluación en el Conjunto de Validación:
   
           * Precisión en el conjunto de validación: 79.28%
           * Pérdida en el conjunto de validación: 0.8031
   
      La implementación de la regularización L2 demostró una mejora en el rendimiento del modelo en comparación con V1. Hubo un incremento en la precisión del conjunto de validación (de 76.15% a 79.28%) y una disminución en la pérdida de validación         (de 0.9511 a 0.8031). Esto indica que la regularización ayudó a reducir el sobreajuste y mejorar la generalización del modelo. Sin embargo, las gráficas de entrenamiento y validación aún muestran cierto grado de sobreajuste.
   
   
   
   ### Test 2: V2 - Strong Regularization (Strength=1e-3, Dropout=0.5)
   
   * En la segunda iteración de V2, se aplicó una regularización L2 más fuerte (strength de 1e-3) y una tasa de dropout aumentada (0.5) con el objetivo de reducir aún más el sobreajuste observado en las pruebas anteriores.
   
   * Gráficas de precisión y pérdida:
   
        ![](https://github.com/maugarciav/tc3002b-IA/blob/main/IMG/TrainValV2-Test2.png)
   
      * Evaluación en el Conjunto de Validación:
      
          * Precisión en el conjunto de validación: 81.09%
          * Pérdida en el conjunto de validación: 0.9722
   
      Los resultados de este test indican una mejora en la precisión del modelo en el conjunto de validación, obteniendo una mejor generalización. Además, las gráficas muestran una menor divergencia entre las curvas de entrenamiento y validación, lo        que confirma una reducción del sobreajuste. Sin embargo, la pérdida en el conjunto de validación *aumentó* a 0.9722 (en comparación con 0.8031 en Test 1). Es decir, el modelo está clasificando más imágenes correctamente, pero la "confianza"           promedio en las predicciones es menor. Esto refleja el efecto de la mayor regularización. En general, este modelo muestra una compensación entre precisión y confianza, con una mejor capacidad para generalizar a expensas de una mayor pérdida.
      
   
   
   ### Test 3: V2 - Moderate Regularization (Strength=5e-4, Dropout=0.45)
   
   * En la tercera iteración de V2, se ajustó la regularización L2 a un strength de 5e-4 y la tasa de dropout a 0.45, buscando un punto intermedio entre las configuraciones de los Test 1 y Test 2.
   
   * Gráficas de precisión y pérdida:
   
       ![](https://github.com/maugarciav/tc3002b-IA/blob/main/IMG/TrainValV2-Test3.png)

   * Matriz de Confusión:
     
       ![](https://github.com/maugarciav/tc3002b-IA/blob/main/IMG/MatrizV2Test3.png)
   
   * Evaluación en el Conjunto de Validación:
   
       * Precisión en el conjunto de validación: 81.91%
       * Pérdida en el conjunto de validación: 0.8296
       * Recall General (promedio): 0.8401
       * F1-Score General (promedio): 0.8283
   
       En este Test el modelo mantiene la alta precisión de validación alcanzada en el Test 2 y, al mismo tiempo, se redujo significativa la pérdida de validación (0.8296). Con estos paramteros se logra un mejor equilibrio entre la capacidad del modelo para generalizar a datos no vistos y la confianza en sus predicciones. Sin embargo, podemos observar en las gráficas fluctuaciones o "saltos" pronunciados, esto podría indicar cierta inestabilidad en el proceso de aprendizaje, lo cual es algo a tomar en cuenta. En fin a comparación de los tests anteriores, el Test 3 nos da los mejores resultados entre precisión y pérdida.

       La matriz de confusión nos dice que, el rendimiento general es bueno pero existen algunas instancias de confusión entre las clases de tumores. Específicamente, entre gliomas y meningiomas. A pesar de estas confusiones, el modelo demuestra una buena sensibilidad general (Recall) y un equilibrio adecuado entre precisión y sensibilidad (F1-Score). Las fluctuaciones observadas en las gráficas de entrenamiento y validación aún indican cierta inestabilidad en el aprendizaje, lo cual es algo que vamos a explorar en siguientes versiones.



##   Versión 3 (V3)

* En V2 las gráficas mostraban fluctuaciones, sugiriendo cierta inestabilidad en el aprendizaje. Para abordar esto y buscar un aprendizaje más estable, en V3 se implementó una arquitectura y técnicas de regularización basadas en el paper "[Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)". El objetivo es reducir las fluctuaciones en las curvas de precisión y pérdida, indicando una mayor estabilidad y confianza en las predicciones.
  
* Los cambios clave con respecto a V2 son:
    * **Dropout Placement:** Se aplicó Dropout después de cada capa MaxPooling2D en las capas convolucionales, además de las capas densas.
    * **Max-Norm Regularization:** Se añadió regularización Max-Norm a la capa Dense intermedia.

* **Técnicas de Regularización Implementadas:**
   * **Dropout:** Esta técnica combate el sobreajuste introduciendo aleatoriedad durante el entrenamiento. En cada paso de entrenamiento, para cada capa donde se aplica, cada neurona tiene una probabilidad `p` de ser retenida (activada) o `1-p` de ser temporalmente "apagada" o "dropeada" (sus salidas se establecen en cero). Esto significa que en cada batch, se entrena una red neuronal "adelgazada" diferente, con una subred distinta de neuronas activas.
   * **Regularización Max-Norm:** Es otra técnica para controlar la complejidad del modelo y prevenir el sobreajuste. Limita la magnitud del vector de pesos entrantes a cada neurona. Esto evita que los pesos crezcan demasiado. Limitar la norma de los pesos restringe el espacio de parámetros y ayuda a la generalización. Se utilizó un valor de `c=3` para la capa densa intermedia.

* La arquitectura que propoone el paper es la siguiente (los valores de dropout pueden variar e irse ajustando para buscar un mejor resultado):

    * **Capas Convolucionales:**
        * `Conv2D` (32 filtros, (3x3), ReLU)
        * `MaxPooling2D` ((2x2))
        * `Dropout`
        * `Conv2D` (64 filtros, (3x3), ReLU)
        * `MaxPooling2D` ((2x2))
        * `Dropout`
        * `Conv2D` (128 filtros, (3x3), ReLU)
        * `MaxPooling2D` ((2x2))
        * `Dropout`
    * **Capas Densas:**
        * `Flatten`
        * `Dropout`
        * `Dense` (256 neuronas, ReLU, **MaxNorm constraint = 3**)
        * `Dropout`
        * `Dense` (num\_classes, Softmax)
     * **Entrenamiento:**
        * Optimizador: Adam (learning rate = 1e-4)
        * Pérdida: Sparse Categorical Crossentropy
        * Métricas: Accuracy

   ### TEST 1: V3 - Increase Dropout
   
   * Para la primera iteración, se ajustaron las tasas de dropout en las capas convolucionales y densas, incrementando el dropout en las primeras capas convolucionales y manteniedno en las capas densas (Esta es la arquitectura mas similar a la propuesta en el paper). La arquitectura es la siguiente:
 
      * **Capas Convolucionales:**
           * `Conv2D` (32 filtros, (3x3), ReLU)
           * `MaxPooling2D` ((2x2))
           * `Dropout` (0.1)
           * `Conv2D` (64 filtros, (3x3), ReLU)
           * `MaxPooling2D` ((2x2))
           * `Dropout` (0.25)
           * `Conv2D` (128 filtros, (3x3), ReLU)
           * `MaxPooling2D` ((2x2))
           * `Dropout` (0.25)
      * **Capas Densas:**
           * `Flatten`
           * `Dropout` (0.5)
           * `Dense` (256 neuronas, ReLU, **MaxNorm constraint = 3**)
           * `Dropout` (0.5)
           * `Dense` (num\_classes, Softmax)
   
   * Gráficas de precisión y pérdida:
   
       ![](https://github.com/maugarciav/tc3002b-IA/blob/main/IMG/TrainValV3-Test1.png)
   
   * Evaluación en el Conjunto de Validación:
   
       * Precisión en el conjunto de validación: 74.01%
       * Pérdida en el conjunto de validación: 0.6699
   
   
      Los resulrado muestran una mejora bastante significativa en la perdida en comparacion con V1 y V2. Esta disminución en la pérdida nos indica que el modelo V3 está realizando predicciones con mayor confianza. Además, tenemos un aprendizaje mucho       más controlado y estable durante el entrenamiento. Sin embargo tambien tenemos una disminucion en la precisión. En general, V3 demuestra que la aplicación de las técnicas de Dropout y Max-Norm, tiene un impacto significativo en la                     regularización del modelo, ayudando a un aprendizaje más estable y mejorando la confianza en las predicciones, pero debemos buscar hiperparametros que no comprmoetan tanto la precisión.
      


   ### TEST 2: V3 - Dropout Variation 
   
   * En la segunda iteración, se redujeron las tasas de dropout en todas las capas. Pues los resultados del Test 1 indican que el modelo, con una regularización más fuerte, estaba generalizando en exceso, lo que comprometía la precisión, La arquitectura es la siguiente:

       * **Capas Convolucionales:**
           * `Conv2D` (32 filtros, (3x3), ReLU)
           * `MaxPooling2D` ((2x2))
           * `Dropout` (0.2)
           * `Conv2D` (64 filtros, (3x3), ReLU)
           * `MaxPooling2D` ((2x2))
           * `Dropout` (0.2)
           * `Conv2D` (128 filtros, (3x3), ReLU)
           * `MaxPooling2D` ((2x2))
           * `Dropout` (0.2)
       * **Capas Densas:**
           * `Flatten`
           * `Dropout` (0.4)
           * `Dense` (256 neuronas, ReLU, **MaxNorm constraint = 3**)
           * `Dropout` (0.4)
           * `Dense` (num\_classes, Softmax)
   
   * Gráficas de precisión y pérdida:
   
       ![](https://github.com/maugarciav/tc3002b-IA/blob/main/IMG/TrainValV3-Test2.png)

  * Matriz de Confusión:

    ![](https://github.com/maugarciav/tc3002b-IA/blob/main/IMG/MatrizV3Test2.png)
   
   * Evaluación en el Conjunto de Validación:
   
       * Precisión en el conjunto de validación: 80.10%
       * Pérdida en el conjunto de validación: 0.5756
       * Recall General (promedio): 0.8109
       * F1-Score General (promedio): 0.7945
   
      La reducción en las tasas de dropout mejoró significativamente la precisión del modelo en comparación con el Test 1, alcanzando un 80.10%. Este aumento en la precisión indica que la menor regularización permitió al modelo capturar información         más relevante de los datos. Si bien esta precisión es ligeramente inferior a la mejor precisión obtenida en V2, la pérdida en el conjunto de validación disminuyó notablemente a 0.5776. En resumen, se observa una compensación: Test 2 ofrece            una mejora considerable en la confianza de las predicciones (menor pérdida) y un aumento en la precisión respecto a Test 1, a pesar de que existe una ligera diferencia con la mejor version de V2, la difreencia en la pérdida es mucho mayor con         V2 Test 3.

      La matriz de confusión nos indica un rendimiento relativamente equilibrado entre las clases, aunque se hay una mayor confusión en la clasificación del meningioma. La clase normal sigue presentando un recall  alto. Los valores generales de Recall (0.8109) y F1-Score (0.7945) indican una buena capacidad del modelo para identificar los casos positivos y un equilibrio razonable entre precisión y sensibilidad. En resumen, Test 2 nos da una mejora considerable en la confianza de las predicciones (menor pérdida) y un aumento en la precisión respecto a Test 1, sin embargo hay una ligera diferencia en presision y F1 con la mejor versión de V2, la diferencia en la pérdida es mucho mayor con V2 Test 3.


## Versión 4 (V4)

   * En esta iteración, se buscó una arquitectura que combinara elementos de las estrategias de regularización exploradas en el paper de Ying y el paper de Srivastava. (Dropout y Max-Norm), con el objetivo de alcanzar un equilibrio óptimo entre una alta precisión y una baja pérdida de validación, indicando una buena generalización y confianza en las predicciones. Después de diferentes pruebas los mejores reultados se llegaron con la siguiente arquitectura:

      * **Capas Convolucionales:**
           * `Conv2D` (32 filtros, (3x3), ReLU, **Regularizador L2: 1e-5**)
           * `MaxPooling2D` ((2x2))
           * `Dropout` (0.15)
           * `Conv2D` (64 filtros, (3x3), ReLU, **Regularizador L2: 1e-5**)
           * `MaxPooling2D` ((2x2))
           * `Dropout` (0.25)
           * `Conv2D` (128 filtros, (3x3), ReLU, **Regularizador L2: 1e-5**)
           * `MaxPooling2D` ((2x2))
           * `Dropout` (0.25)
       * **Capas Densas:**
           * `Flatten`
           * `Dropout` (0.3)
           * `Dense` (256 neuronas, ReLU, **MaxNorm constraint = 3**)
           * `Dropout` (0.3)
           * `Dense` (num\_classes, Softmax)
       * **Entrenamiento:**
           * Optimizador: Adam (learning rate = 1e-4)
           * Pérdida: Sparse Categorical Crossentropy
           * Métricas: Accuracy

   
   * Gráficas de precisión y pérdida:
   
       ![](https://github.com/maugarciav/tc3002b-IA/blob/main/IMG/TrainValV4.png)

   * Matriz de Confusión:

       ![](https://github.com/maugarciav/tc3002b-IA/blob/main/IMG/MatrizV4.png)

   
   * Evaluación en el Conjunto de Validación:
   
       * Precisión en el conjunto de validación: 80.26%
       * Pérdida en el conjunto de validación: 0.6187
       * Recall General: 0.8026
       * F1-Score General: 0.8001

      Si bien esta arquitectura híbrida no logró superar la mejor precisión obtenida en pruebas anteriores (81.91% en V2 - Test 3), sí demostró un **excelente equilibrio entre la precisión (80.26%) y la pérdida de validación (0.6187)**. Esta baja           pérdida sugiere que el modelo realiza predicciones con una alta confianza, a la vez que mantiene una capacidad de generalización robusta gracias a la combinación de las diferentes técnicas de regularización.
     
      Al igual que en las versiones anteriores (V2 y V3), la matriz de confusión de V4 revela un buen rendimiento general en la clasificación, con la clase normal presentando un recall alto, lo que indica una buena capacidad para identificar correctamente los escáneres sin tumores. Sin embargo, al igual que en los modelos previos, se observa una tendencia a la confusión entre glioma y meningioma. A pesar de esta dificultad persistente, los valores generales de Recall y F1-Score indican un buen rendimiento.


## Conclusiones

Resumen de resultados de los mejores test por verisón.

| Modelo      | Precisión en Validación | Pérdida en Validación | Recall General | F1-Score General |
| :---------- | :---------------------: | :--------------------: | :------------: | :--------------: |
| V2 - Test 3 |         81.91%          |         0.8296         |     0.8401     |      0.8283      |
| V3 - Test 2 |         80.10%          |         0.5756         |     0.8109     |      0.7945      |
| V4          |         80.26%          |         0.6187         |     0.8026     |      0.8001      |

**Análisis de los Resultados:**
* **Precisión:** V2 - Test 3 obtuvo la mayor precisión en el conjunto de validación (81.91%).
* **Pérdida:** V3 - Test 2 registró la menor pérdida en el conjunto de validación (0.5756).
* **Recall General:** V2 - Test 3 también obtuvo el mejor Recall general (0.8401).
* **F1-Score General:** V2 - Test 3 lideró también en el F1-Score general (0.8283).


Si bien V2 - Test 3 logró la mayor precisión, V3 - Test 2 demostró la menor pérdida de validación, lo que sugiere una mayor confianza en sus predicciones. V4, diseñado como una combinación de estrategias, ofreció un rendimiento equilibrado en todas las métricas.

Es importante notar que, a pesar de las diferencias en las métricas generales, el análisis de las matrices de confusión reveló una tendencia común en los tres modelos: un buen rendimiento en la identificación de escáneres normales (alto recall para la clase normal) y una mayor dificultad para distinguir entre gliomas y meningiomas, lo que podría ser un área para futuras investigaciones y mejoras en el modelo.

**Considerando la Importancia de la Detección de Tumores:**

En el contexto de la detección de tumores cerebrales, **minimizar los falsos negativos es muy importante**. Un falso negativo (decirle a alguien que no tiene un tumor cuando sí lo tiene) puede retrasar el diagnóstico y el tratamiento, con consecuencias potencialmente graves para la salud del paciente. Por otro lado, un falso positivo (decirle a alguien que tiene un tumor cuando no lo tiene) puede generar ansiedad y requerir pruebas adicionales innecesarias, lo cual también tiene un costo emocional y económico. Sin embargo, **el impacto de un falso negativo suele ser considerado más grave**. Por lo tanto, un modelo con un **Recall más alto** es preferible, ya que indica una mayor capacidad para detectar todos los casos positivos reales.

**Elección del Mejor Modelo:**

Basándonos en esta consideración, **V2 - Test 3 parece ser el modelo más adecuado**. Aunque no tiene la menor pérdida, presenta el **Recall general más alto (0.8401)**, lo que significa que es el más efectivo para minimizar la probabilidad de no detectar un tumor presente. Su alta precisión y F1-Score también respaldan su buen rendimiento general.

Si bien V3 - Test 2 muestra la menor pérdida (mayor confianza en las predicciones), su Recall es ligeramente inferior. V4 ofrece un buen equilibrio, pero su Recall tampoco supera al de V2 - Test 3.

**Conclusión Final:**

Dado el contexto de la detección de tumores cerebrales y la mayor importancia de minimizar los falsos negativos, **V2 - Test 3 es la opción preferible debido a su Recall general más alto**. 



## Ejecutar localmente

Sigue estos pasos:

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/maugarciav/tc3002b-IA.git
    cd tc3002b-IA
    ```

2.  **Descargar los modelos pre-entrenados:**
    ```bash
    Debido al tamaño de los archivos de los modelos pre-entrenados, no se incluyen en el repositorio.
    Deberás descargar los modelos desde el siguiente enlace de Google Drive: https://drive.google.com/drive/folders/1m6FYst7gtG0msE0miAY-0gTY1NLh4EjI?usp=sharing

    Encontrarás tres archivos:
        - modelo_deteccion_tumores_V2TEST3.keras
        - modelo_deteccion_tumores_V3TEST2.keras
        - modelo_deteccion_tumores_V4.keras

    Descarga los tres archivos.
    ```

3.  **Configurar el modelo a utilizar en `app.py`:**
    ```bash
    1. Abre el archivo `web_tumor_detector/app.py` con un editor de texto.
    2. Localiza la siguiente línea de código:

           ```python
           model = tf.keras.models.load_model('') #!!!NOMBRE DEL MODELO QUE QUIERS PROBAR!!!!!
           ```

    3.  Escribe el nombre del archivo del modelo que deseas utilizar (asegurándote de incluir la extensión `.keras`). Por ejemplo, si quieres usar el modelo V4, la línea debería quedar así:

           ```python
           model = tf.keras.models.load_model('modelo_deteccion_tumores_V4.keras') #!!!NOMBRE DEL MODELO QUE QUIERS PROBAR!!!!!
           ```

    4.  Guarda los cambios en el archivo `app.py`.
    ```

4.  **Crear y activar el entorno virtual:**
    ```bash
    cd web_tumor_detector
    python3 -m venv venv
    source venv/bin/activate     # En macOS y Linux
    venv\Scripts\activate.bat   # En Windows
    ```

5.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

6.  **Ejecutar la aplicación:**
    ```bash
    python app.py
    ```

7.  **Abrir en el navegador:**
    Ve a `http://127.0.0.1:5000/` en tu navegador.

## Autor

Mauricio Garcia Villanuvea


## Referencias

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine Learning Research, 15, 1929-1958, https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf

Ying, X. (2019). An Overview of Overfitting and its Solutions. IOP Conference Series: Journal of Physics: Conference Series, 1168, 022022. https://iopscience.iop.org/article/10.1088/1742-6596/1168/2/022022/pdf


## Licencia

[Licencia del proyecto](https://creativecommons.org/publicdomain/zero/1.0/)
