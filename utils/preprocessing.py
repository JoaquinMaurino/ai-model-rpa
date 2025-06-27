import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np # Importar numpy para manejo de NaN

def preparar_dataset_entrenamiento(filas):
    """
    Prepara el dataset de entrenamiento para modelos de ML.
    Realiza One-Hot Encoding en columnas categóricas y Min-Max Scaling en numéricas.

    Args:
        filas (list): Lista de tuplas obtenidas de la consulta a la base de datos.

    Returns:
        tuple: (X_train, X_test, y_train, y_test), preprocessor_pipeline, label_encoder_y
               Donde preprocessor_pipeline es el ColumnTransformer/Pipeline ajustado.
    """
    # Define las columnas exactamente como son devueltas por get_dataset_entrenamiento()
    # ¡Nombres de columnas actualizados según la última versión de data.py!
    df = pd.DataFrame(filas, columns=[
        "id_producto",
        "producto_categoria",
        "id_deposito_asignado", # Este es el TARGET (variable objetivo) ahora
        "deposito_categorias_aceptadas",
        "deposito_capacidad_maxima_producto", # Nueva columna
        "deposito_capacidad_minima_producto", # Nueva columna
        "stock_actual_producto_en_deposito",  # Nueva columna
        "total_vendido_producto_deposito_historico"
    ])

    # --- Pasos de Preprocesamiento ---

    # 1. Manejar 'deposito_categorias_aceptadas':
    # Asegurarse de que sea tratada como una cadena para una codificación One-Hot consistente.
    df["deposito_categorias_aceptadas"] = df["deposito_categorias_aceptadas"].astype(str)

    # Identificar columnas categóricas y numéricas para el preprocesamiento
    # ¡Actualizadas con las nuevas columnas!
    categorical_features = ["producto_categoria", "deposito_categorias_aceptadas"]
    numerical_features = [
        "deposito_capacidad_maxima_producto",      # Nueva numérica
        "deposito_capacidad_minima_producto",      # Nueva numérica
        "stock_actual_producto_en_deposito",       # Nueva numérica
        "total_vendido_producto_deposito_historico"
    ]

    # --- Importante: Manejo de valores nulos ---
    # Las columnas 'deposito_capacidad_maxima_producto' y 'deposito_capacidad_minima_producto'
    # pueden tener NaN si la combinación producto-deposito no existía en Deposito_Capacidad_Producto.
    # Así mismo, 'stock_actual_producto_en_deposito' puede ser 0 pero no NaN si viene de COALESCE.
    # Para el prototipo, un enfoque simple es rellenar NaN con un valor por defecto (ej. 0 o la media/mediana).
    # Como estas representan capacidades, 0 o un valor muy pequeño podría ser razonable.
    # Para capacidades (min/max), un 0 tiene sentido si no hay regla específica.
    # Para stock, ya viene como 0 por COALESCE.
    for col in ["deposito_capacidad_maxima_producto", "deposito_capacidad_minima_producto"]:
        df[col] = df[col].fillna(0) # Asume 0 si no hay una capacidad definida explícitamente

    # Separar características (X) y variable objetivo (y)
    # ¡id_deposito_asignado es ahora la variable objetivo!
    X = df.drop(columns=["id_producto", "id_deposito_asignado"]) 
    y = df["id_deposito_asignado"] # El depósito asignado históricamente (tu target)

    # Codificar la variable objetivo 'y' usando LabelEncoder
    label_encoder_y = LabelEncoder()
    y_encoded = label_encoder_y.fit_transform(y)

    # Crear un ColumnTransformer para el preprocesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

    # Ajustar y transformar las características
    X_processed = preprocessor.fit_transform(X)

    # Convertir X procesado de nuevo a un DataFrame para mejor legibilidad y nombres de columna
    # Obtener los nombres de las características después de One-Hot Encoding
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(ohe_feature_names)
    
    X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed_df, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded 
    )

    return (X_train, X_test, y_train, y_test), preprocessor, label_encoder_y




def preparar_datos_para_prediccion(df_nuevos_datos: pd.DataFrame, preprocessor_ajustado: ColumnTransformer) -> pd.DataFrame:
    """
    Prepara nuevos datos (DataFrame) para ser usados en la predicción,
    utilizando un preprocesador ColumnTransformer ya ajustado.

    Args:
        df_nuevos_datos (pd.DataFrame): DataFrame con las nuevas características
                                        (sin la columna de ID de depósito original).
        preprocessor_ajustado (ColumnTransformer): El ColumnTransformer ya ajustado
                                                   durante el entrenamiento.

    Returns:
        pd.DataFrame: DataFrame con las características preprocesadas, listo para el modelo.
    """
    # Asegurarse de que 'deposito_categorias_aceptadas' sea tratada como string
    if "deposito_categorias_aceptadas" in df_nuevos_datos.columns:
        df_nuevos_datos["deposito_categorias_aceptadas"] = df_nuevos_datos["deposito_categorias_aceptadas"].astype(str)

    # Nombres de las características categóricas y numéricas deben coincidir con el entrenamiento
    categorical_features = ["producto_categoria", "deposito_categorias_aceptadas"]
    numerical_features = [
        "deposito_capacidad_maxima_producto",
        "deposito_capacidad_minima_producto",
        "stock_actual_producto_en_deposito",
        "total_vendido_producto_deposito_historico"
    ]

    # Asegurar que todas las columnas esperadas por el preprocesador estén presentes
    # y rellenar NaN si es necesario (aunque ya se hace en get_deposito_candidato_features)
    for col in numerical_features:
        if col not in df_nuevos_datos.columns:
            df_nuevos_datos[col] = 0 # O un valor por defecto apropiado si la columna falta
        df_nuevos_datos[col] = df_nuevos_datos[col].fillna(0) # Doble check para NaN

    # Transformar los datos usando el preprocesador ya ajustado
    # No usar .fit_transform() aquí, solo .transform()
    X_pred_processed = preprocessor_ajustado.transform(df_nuevos_datos)

    # Obtener los nombres de las características después de One-Hot Encoding
    ohe_feature_names = preprocessor_ajustado.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(ohe_feature_names)
    
    # Convertir a DataFrame para consistencia y para que el modelo lo espere
    X_pred_df = pd.DataFrame(X_pred_processed, columns=all_feature_names)
    
    return X_pred_df