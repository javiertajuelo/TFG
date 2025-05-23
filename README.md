# TFG
Trabajo Fin de Grado en Ingeniería Informática, Universidad Complutense de Madrid.

## Contenido

### Resumen
En este proyecto se estudia el campo de los sistemas de recomendación multidominio, con el objetivo de estudiar, analizar y comparar los resultados de rendimiento obtenidos entre los modelos de recomendación propuestos. Para lograrlo, se proponen modelos multidominios basados en contenido y en filtrado colaborativo con diferentes técnicas. 

Además, también se implementan sistemas de recomendación sobre los modelos y los códigos para el análisis y acondicionamiento del conjunto de datos utilizado en la evaluación de los modelos. 

### Abstract

In this project, we explore the field of multidomain recommendation systems with the goal of examining, analyzing, and comparing the performance results achieved by the proposed recommendation models. To this end, we introduce multidomain models based on content and collaborative filtering using various techniques.

Additionally, we implement recommendation systems for these models and provide the code for analyzing and preprocessing the dataset used in the models’ evaluation.

### Estructura

#### Acondicionamiento Datasets de Ratings
Contiene los scripts y notebooks encargados de limpiar y preparar las interacciones usuario–ítem. Aquí se aplica el filtrado 5-core para garantizar al menos cinco valoraciones por usuario, se unifican formatos, se eliminan registros inválidos y se generan los ficheros de entrenamiento, validación y prueba en formatos CSV y pickle para su uso posterior.

#### Acondicionamiento MetaData
Incluye el código de preprocesamiento de la información de producto (títulos, categorías, descripciones, precios). Se limpian campos nulos, se normaliza texto (tokenización, stop-words, lematización) y se exportan las matrices de atributos (por ejemplo, TF-IDF o Count Vectorizer) que servirán como entrada para los modelos basados en contenido.

#### Análisis DataSets
Notebooks de análisis exploratorio donde se estudian estadísticas descriptivas (número de usuarios, ítems, densidad, distribución de ratings por dominio), visualizaciones de patrones de uso y comparación entre dominios (películas, libros, música). Sirve para entender las características y desafíos de cada conjunto antes de modelar.

#### Aproximaciones
Incluye los modelos de recomendación más básicos propuestos en el proyecto.

#### Aumento de Densidad de los Datasets
Scripts que exploran técnicas de enriquecimiento de datos para mitigar la dispersión:
- Muestreo de interacciones negativas.  
- Transferencia de interacciones cruzando dominios.  
- Generación sintética de preferencias.  
Permiten evaluar cómo afectan estas estrategias al rendimiento de los modelos.

#### DataSets
Incluye los conjuntos de datos usados en la evaluación de los modelos.

#### EDDA
Modelo de recomendación EDDA. Incluye tanto el modelo de entrenamiento como la implementación de la recomendación para dicho modelo.

## Autores
- Jaime García Redondo – [Jaigar15](https://github.com/Jaigar15)
- Javier Tajuelo Moreno-Palancas – [javiertajuelo](https://github.com/javiertajuelo)
- Iván Ochoa Plaza - [Iochoa01](https://github.com/Iochoa01)

