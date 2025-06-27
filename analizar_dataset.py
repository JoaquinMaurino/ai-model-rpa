import pandas as pd
import matplotlib.pyplot as plt
from db.data import get_dataset_entrenamiento

def analizar_dataset():
    datos = get_dataset_entrenamiento()
    if not datos:
        print("❌ No hay datos para analizar.")
        return
    
    # Defino columnas según tu dataset
    columnas = [
        "nombre", "categoria", "color", "talle",
        "id_deposito", "categorias_deposito",
        "capacidad_maxima", "capacidad_minima",
        "deposito_asignado"
    ]
    
    df = pd.DataFrame(datos, columns=columnas)
    
    print("\n--- INFORMACIÓN GENERAL ---")
    print(f"Total filas: {len(df)}")
    print("\nPrimeras filas:")
    print(df.head())
    print("\nTipos de datos:")
    print(df.dtypes)
    
    print("\n--- VALORES NULOS POR COLUMNA ---")
    print(df.isnull().sum())
    
    print("\n--- DISTRIBUCIÓN DE LA VARIABLE TARGET (deposito_asignado) ---")
    print(df["deposito_asignado"].value_counts())
    
    plt.figure(figsize=(10,5))
    df["deposito_asignado"].value_counts().plot(kind='bar', title="Distribución depósitos asignados")
    plt.xlabel("ID Depósito")
    plt.ylabel("Cantidad de ingresos")
    plt.show()
    
    print("\n--- VALORES ÚNICOS EN VARIABLES CATEGÓRICAS ---")
    cols_categoricas = ["nombre", "categoria", "color", "talle", "categorias_deposito"]
    for col in cols_categoricas:
        print(f"{col}: {df[col].nunique()} valores únicos")
    
    print("\n--- NÚMERO DE DEPÓSITOS POR CATEGORÍA DE PRODUCTO ---")
    dep_por_cat = df.groupby("categoria")["deposito_asignado"].nunique()
    print(dep_por_cat)
    
    plt.figure(figsize=(10,5))
    dep_por_cat.plot(kind='bar', title="Cantidad depósitos únicos por categoría")
    plt.xlabel("Categoría")
    plt.ylabel("Cantidad depósitos")
    plt.show()
    
    print("\n--- ANÁLISIS DE COHERENCIA ---")
    print("Chequeá que las categorías del producto correspondan a depósitos que aceptan esa categoría (debes hacerlo con datos complementarios).")
    print("Si esta relación está mal definida, el modelo no podrá aprender correctamente.")
    
    print("\n--- RESUMEN ---")
    if len(df) < 500:
        print("⚠️ El dataset es pequeño, considera aumentar datos.")
    if df["deposito_asignado"].nunique() > 20:
        print("⚠️ Hay muchas clases para la variable target; esto puede dificultar el aprendizaje.")
    if any(df.isnull().sum() > 0):
        print("⚠️ Hay valores nulos que deberías imputar o eliminar.")
    print("Si la distribución de clases está muy desequilibrada, considera técnicas de balanceo.")
    print("Si las variables categóricas tienen muchísimos valores únicos, podrías agrupar o reducir cardinalidad.")

if __name__ == "__main__":
    analizar_dataset()
