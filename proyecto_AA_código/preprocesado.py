"""Limpieza y preprocesamiento del dataset del proyecto de diabetes.

Lo tengo en un archivo aparte (y no solo en el notebook) porque estas mismas funciones
las vamos a necesitar más adelante en la API: los datos nuevos que mande el profesor
tienen que pasar exactamente por la misma limpieza y el mismo preprocesamiento que los
datos con los que se entrenaron los modelos.
"""

__author__ = "Pablo González Ferro"

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------------------------
# Constantes que salen del EDA
# ---------------------------------------------------------------------------

# Columnas que se eliminan: datos personales ofuscados y la columna de gestión
COLS_PII = ["dni", "nombre", "apellidos", "direccion", "telefono", "email", "num_historia_clinica"]
COLS_DESCARTAR = COLS_PII + ["usuario_alta_datos"]
COL_ID = "id_paciente_proy"

# Variables objetivo de las dos líneas supervisadas
OBJ_CLF = "diabetes"
OBJ_REG = "dias_salud_fisica_mala"

# Columnas que deberían ser si/no
COLS_BIN = ["presion_alta", "colesterol_alto", "chequeo_colesterol_5a", "fumador", "ictus",
            "enfermedad_cardiaca", "actividad_fisica", "consume_fruta", "consume_verdura",
            "consumo_alcohol_alto", "dificultad_caminar", "diabetes"]

# Todas las formas en las que aparece un si/no en el archivo, traducidas a 1/0 (se comparan en minúsculas)
MAPA_BIN = {"si": 1, "true": 1, "1.0": 1, "1": 1, "no": 0, "false": 0, "0.0": 0, "0": 0}

# Orden de cada variable ordinal, de menor a mayor (1 = la primera categoría de la lista)
ORDINALES = {
    "salud_general": ["excelente", "muy buena", "buena", "regular", "mala"],
    "grupo_edad": ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
                   "55-59", "60-64", "65-69", "70-74", "75-79", "80+"],
    "nivel_educativo": ["sin estudios", "primaria", "secundaria incompleta", "secundaria",
                        "formacion superior parcial", "titulado superior"],
    "nivel_ingresos": ["<10k", "10-15k", "15-20k", "20-25k", "25-35k", "35-50k", "50-75k", ">75k"],
}

# Las tres variables numéricas
NUMERICAS = ["imc", "dias_salud_mental_mala", "dias_salud_fisica_mala"]


# ---------------------------------------------------------------------------
# 1. Limpieza: no aprende nada de los datos, así que vale igual para train, test o la API
# ---------------------------------------------------------------------------

def limpiar(crudo):
    """Aplica la limpieza del EDA a un dataframe con el formato del CSV original.

    Funciona tanto si el CSV se ha leído todo como texto como si se ha leído normal
    (con NaN), y no falla si falta alguna columna, por ejemplo las variables objetivo
    cuando lleguen datos nuevos a la API.
    """
    df = crudo.copy()

    # El id pasa a ser el índice (si viene)
    if COL_ID in df.columns:
        df[COL_ID] = pd.to_numeric(df[COL_ID])
        df = df.set_index(COL_ID)

    # Quito los datos personales y la columna de gestión que vengan
    df = df.drop(columns=[c for c in COLS_DESCARTAR if c in df.columns])

    # Paso todo a texto sin espacios, y tanto las celdas vacías como los NaN quedan como NaN
    df = df.apply(lambda s: s.astype(str).str.strip().where(s.notna(), np.nan))
    df = df.replace({"": np.nan, "nan": np.nan, "None": np.nan})

    # Binarias: todas las variantes (si/SI/True/1.0...) a 1/0
    for c in [c for c in COLS_BIN if c in df.columns]:
        df[c] = df[c].str.lower().map(MAPA_BIN).astype("Int64")

    # sexo: hombre = 1, mujer = 0
    if "sexo" in df.columns:
        df["sexo"] = df["sexo"].str.lower().map({"hombre": 1, "mujer": 0}).astype("Int64")

    # Ordinales: cada categoría a su posición (1, 2, 3...). Un valor desconocido queda como NaN
    for c, orden in ORDINALES.items():
        if c in df.columns:
            df[c] = df[c].str.lower().map({v: i + 1 for i, v in enumerate(orden)}).astype("Int64")

    # Numéricas a número (si algo no se puede convertir, queda como NaN)
    for c in [c for c in NUMERICAS if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# 2. Preprocesamiento: sí aprende de los datos, así que se ajusta SOLO con el train
# ---------------------------------------------------------------------------

def ajustar_preprocesador(X_train):
    """Ajusta los imputadores y el escalador con el conjunto de entrenamiento.

    - Numéricas (imc y conteos de días): imputación con la mediana, porque son asimétricas.
    - Binarias y ordinales: imputación con la moda.
    - Escalado con RobustScaler de las columnas que no son binarias, por los atípicos del IMC.
      Las binarias ya están en 0/1 y no hace falta tocarlas.

    Devuelve un diccionario con todo lo ajustado, que es lo que se guarda como artefacto.
    """
    cols_num = [c for c in X_train.columns if c in NUMERICAS]
    cols_cat = [c for c in X_train.columns if c not in cols_num]
    cols_escalar = [c for c in X_train.columns if c in NUMERICAS or c in ORDINALES]

    # Imputadores, cada uno con su estrategia, ajustados solo con el train
    imp_num = SimpleImputer(strategy="median").fit(X_train[cols_num].astype(float)) if cols_num else None
    imp_cat = SimpleImputer(strategy="most_frequent").fit(X_train[cols_cat].astype(float)) if cols_cat else None

    prep = {
        "columnas": list(X_train.columns),
        "cols_num": cols_num,
        "cols_cat": cols_cat,
        "cols_escalar": cols_escalar,
        "imp_num": imp_num,
        "imp_cat": imp_cat,
        "escalador": None,
    }

    # El escalador se ajusta sobre el train ya imputado (RobustScaler no admite nulos)
    X_imp = aplicar_preprocesador(prep, X_train, escalar=False)
    prep["escalador"] = RobustScaler().fit(X_imp[cols_escalar]) if cols_escalar else None
    return prep


def aplicar_preprocesador(prep, X, escalar=True):
    """Aplica a X lo que se ajustó con el train: imputa y, si escalar=True, escala.

    Devuelve un dataframe con las mismas columnas y en el mismo orden que el train.
    """
    # Me aseguro de tener las mismas columnas y en el mismo orden que en el train
    X = X.reindex(columns=prep["columnas"]).astype(float)
    salida = X.copy()

    # Imputación
    if prep["imp_num"] is not None:
        salida[prep["cols_num"]] = prep["imp_num"].transform(X[prep["cols_num"]])
    if prep["imp_cat"] is not None:
        salida[prep["cols_cat"]] = prep["imp_cat"].transform(X[prep["cols_cat"]])

    # Escalado
    if escalar and prep["escalador"] is not None:
        salida[prep["cols_escalar"]] = prep["escalador"].transform(salida[prep["cols_escalar"]])
    return salida
