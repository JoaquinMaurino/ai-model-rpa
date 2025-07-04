import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from db.data import get_dataset_entrenamiento
from utils.preprocessing import preparar_dataset_entrenamiento
import joblib

# Importar los modelos de Scikit-learn
from sklearn.ensemble import RandomForestClassifier # Para Random Forest
from sklearn.neighbors import KNeighborsClassifier # Para K-Nearest Neighbors (opcional)
from sklearn.metrics import classification_report, accuracy_score

def train_and_save_model():
    """
    Obtiene los datos, los preprocesa, entrena un modelo de clasificación
    (Random Forest por defecto), lo evalúa y lo guarda junto con el preprocesador
    y el LabelEncoder.
    """
    print("🔄 Obteniendo datos de la base de datos...")
    datos = get_dataset_entrenamiento()
    if not datos:
        print("❌ No hay datos para entrenar. Asegúrate de que la base de datos no esté vacía.")
        return

    print("📊 Preprocesando datos...")
    # Desempaquetar correctamente la salida de la función preparar_dataset_entrenamiento
    (X_train, X_test, y_train, y_test), preprocessor, label_encoder_y = preparar_dataset_entrenamiento(datos)

    print(f"Dimensiones de X_train: {X_train.shape}")
    print(f"Dimensiones de y_train: {y_train.shape}")
    print(f"Dimensiones de X_test: {X_test.shape}")
    print(f"Dimensiones de y_test: {y_test.shape}")


    # --- Selección y Entrenamiento del Modelo ---

    # Opción 1: Random Forest Classifier (Recomendado)
    print("🧠 Entrenando modelo Random Forest Classifier...")
    # n_estimators: número de árboles en el bosque
    # random_state: para reproducibilidad de resultados
    # class_weight: útil si tus clases (depósitos) están desbalanceadas
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print("✅ Entrenamiento de Random Forest completado.")

    # Opción 2: K-Nearest Neighbors Classifier (Descomentar para usar)
    # print("🧠 Entrenando modelo K-Nearest Neighbors Classifier...")
    # model = KNeighborsClassifier(n_neighbors=5) # n_neighbors: número de vecinos a considerar
    # model.fit(X_train, y_train)
    # print("✅ Entrenamiento de K-NN completado.")


    # --- Evaluación del Modelo ---
    print("📈 Evaluando el modelo en el conjunto de prueba...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuracy en el conjunto de prueba: {accuracy:.4f}")

    # Mostrar reporte de clasificación para una evaluación más detallada
    # Reconvertimos las etiquetas numéricas a los nombres originales para el reporte
    y_test_original = label_encoder_y.inverse_transform(y_test)
    y_pred_original = label_encoder_y.inverse_transform(y_pred)

    print("\n--- Reporte de Clasificación ---")
    print(classification_report(y_test_original, y_pred_original))


    # --- Guardado del Modelo y Componentes de Preprocesamiento ---
    save_dir = Path("models/saved")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Guardar el modelo entrenado
    model_path = save_dir / "modelo_deposito.joblib"
    joblib.dump(model, model_path)
    print(f"✅ Modelo guardado en: {model_path}")

    # Guardar el preprocesador (ColumnTransformer)
    preprocessor_path = save_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"✅ Preprocesador guardado en: {preprocessor_path}")

    # Guardar el LabelEncoder de la variable objetivo (y)
    label_encoder_y_path = save_dir / "label_encoder_y.joblib"
    joblib.dump(label_encoder_y, label_encoder_y_path)
    print(f"✅ LabelEncoder para 'y' guardado en: {label_encoder_y_path}")

    print("\n📦 Todos los componentes del modelo (modelo, preprocesador, LabelEncoder) han sido guardados.")

    # --- Visualización (Adaptada para modelos de Scikit-learn) ---
    # Los modelos de Scikit-learn no tienen un "history" como Keras para graficar
    # la pérdida y la precisión por época. Sin embargo, podemos mostrar métricas
    # importantes como la importancia de las características para Random Forest.

    if isinstance(model, RandomForestClassifier):
        print("\n--- Importancia de las Características (Random Forest) ---")
        # Asegurarse de que X_train_df tiene los nombres de las columnas correctos
        # X_train es un DataFrame con los nombres de columnas correctos del preprocesamiento
        feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
        feature_importances.nlargest(10).plot(kind='barh')
        plt.title('Importancia de las Características')
        plt.xlabel('Importancia Relativa')
        plt.ylabel('Característica')
        plt.tight_layout()
        plt.show()
    else:
        print("\nNota: La visualización de la historia de entrenamiento no es aplicable directamente a K-NN.")
        print("Para Random Forest, se puede visualizar la importancia de las características.")


if __name__ == "__main__":
    train_and_save_model()