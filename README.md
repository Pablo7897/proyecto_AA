# Proyecto Aprendizaje Automático: diabetes

Proyecto de la asignatura de Aprendizaje Automático (UIE). A partir de la encuesta de salud de un consorcio de atención primaria hay que trabajar tres líneas:

1. **Clusterización**: sacar perfiles de paciente que se puedan explicar en una frase.
2. **Clasificación**: predecir si un paciente tiene `diabetes`.
3. **Regresión**: predecir `dias_salud_fisica_mala` (de 0 a 30).

Aquí explico qué hay en el repo para que sepáis qué es cada cosa. Todo está dentro de la carpeta `proyecto_AA_código`.

## Datos originales

- `proy_diabetes_dev.csv`: el CSV del Campus Virtual tal cual, sin tocar. De aquí sale todo lo demás.

## EDA

- `EDA.ipynb`: el análisis exploratorio. Sigue la estructura de la PLD1 (estructura, estadísticas, distribuciones, relaciones, nulos, atípicos y variables objetivo). Dentro tiene una auditoría de calidad como la de la PLE1, en la que cada problema encontrado se apunta con `registra()`. Al final hay un apartado orientado al clustering y unas conclusiones con las decisiones de preprocesamiento que salen del análisis.
- `incidencias.csv`: la tabla con todos los problemas de la auditoría (columna, tipo de incidencia, filas afectadas y qué se ha hecho). La genera el EDA al ejecutarse.
- `proy_diabetes_limpio.csv`: los datos después de la limpieza, es decir, sin columnas personales, con las binarias en 0/1 y las ordinales en números, pero todavía con nulos. También lo genera el EDA.

## Preprocesamiento

- `preprocesamiento.ipynb`: deja los datos listos para entrenar. Para cada línea separa train y test (80/20, estratificando), rellena los nulos con `SimpleImputer` (mediana o moda) y escala con `RobustScaler`. Todo se ajusta solo con el train para que no haya fuga de información. Al final hay una prueba que simula cómo llegarán los datos nuevos a la API.
- `preprocesado.py`: las funciones de limpieza y preprocesamiento que usa el notebook. Están en un archivo aparte porque la API tendrá que aplicar exactamente lo mismo a los datos nuevos, así que la podremos importar desde ahí.
- `datos_preprocesados/`: los datos ya listos para modelar, con una carpeta por línea:
  - `clasificacion/` y `regresion/`: `train.csv` y `test.csv` sin escalar (para árboles y bosques), y `train_escalado.csv` y `test_escalado.csv` (para regresión logística, KNN o SVM). La variable objetivo es la última columna.
  - `clustering/`: `datos.csv` y `datos_escalado.csv` con las 7 variables clínicas elegidas. Aparte está `variables_descriptivas.csv`, con la diabetes y los días de mala salud, para describir los grupos una vez hechos.
- `artefactos/`: el imputador y el escalador ya ajustados de cada línea, guardados con `joblib`. Los necesitaremos en la API para preprocesar los datos que mande el profesor.

## Cómo usarlo

- Si solo vais a entrenar modelos, no hace falta ejecutar nada: cargad directamente los CSV de `datos_preprocesados/`.
- Si se cambia algo de la limpieza o del preprocesamiento, hay que ejecutar primero `EDA.ipynb` y después `preprocesamiento.ipynb`, en ese orden, porque el segundo comprueba que su limpieza coincide con la del EDA.
- Antes de ponerse a trabajar, haced Pull para tener la última versión y no pisar cambios de nadie.
