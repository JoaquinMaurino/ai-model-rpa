import sys
from pathlib import Path
import joblib
import pandas as pd
from sqlalchemy import text
from db.connection import get_engine

# Importar solo la función necesaria de preprocessing
# Asegúrate de que 'utils/preprocessing.py' tenga una función 'preparar_datos_para_prediccion'
from utils.preprocessing import preparar_datos_para_prediccion

# --- Configuración de Rutas de Archivos Guardados ---
MODEL_PATH = Path('models/saved/modelo_deposito.joblib')
PREPROCESSOR_PATH = Path('models/saved/preprocessor.joblib')
LABEL_ENCODER_Y_PATH = Path('models/saved/label_encoder_y.joblib')

# --- Ajuste de sys.path para importaciones ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Funciones de Carga de Modelo y Componentes ---

def load_model_and_components():
    """Carga el modelo y los objetos de preprocesamiento guardados."""
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        label_encoder_y = joblib.load(LABEL_ENCODER_Y_PATH)
        print("📥 Cargando modelo y componentes de preprocesamiento...")
        print("✅ Modelo y componentes cargados exitosamente.")
        return model, preprocessor, label_encoder_y
    except FileNotFoundError as e:
        print(f"❌ Error al cargar los componentes del modelo: {e}.")
        print(f"Asegúrate de que 'training.py' se haya ejecutado exitosamente y los archivos existan en '{MODEL_PATH.parent}'.")
        return None, None, None
    except Exception as e:
        print(f"❌ Error inesperado al cargar el modelo: {e}")
        return None, None, None

# --- Funciones de Interacción con la Base de Datos ---

def get_todos_los_depositos_info():
    """Obtiene información básica de todos los depósitos desde la base de datos, incluyendo las categorías que aceptan."""
    engine = get_engine()
    if not engine:
        print("❌ No se pudo conectar a la base de datos para obtener información de depósitos.")
        return []
    try:
        with engine.connect() as conn:
            query = text("SELECT id_deposito, categorias FROM deposito")
            result = conn.execute(query).fetchall()
            return [{**row._mapping} for row in result]
    except Exception as e:
        print(f"❌ Error al obtener la información de los depósitos: {e}")
        return []

def get_deposito_candidato_features(id_producto, deposito_id):
    """
    Obtiene las características necesarias de la base de datos para un producto y un depósito candidato.
    Esto simula una fila de datos de entrada para el modelo.
    """
    engine = get_engine()
    if not engine:
        print("❌ No se pudo conectar a la base de datos para obtener features de depósito.")
        return None

    try:
        with engine.connect() as conn:
            query = text("""
                SELECT
                    p.categoria AS producto_categoria,
                    d.categorias AS deposito_categorias_aceptadas,
                    dcp.capacidad_maxima_unidades AS deposito_capacidad_maxima_producto,
                    dcp.capacidad_minima_unidades AS deposito_capacidad_minima_producto,
                    COALESCE(inv.cantidad, 0) AS stock_actual_producto_en_deposito,
                    COALESCE(SUM(vh.cantidad), 0) AS total_vendido_producto_deposito_historico
                FROM
                    producto p
                JOIN
                    deposito d ON d.id_deposito = :deposito_id
                LEFT JOIN
                    deposito_capacidad_producto dcp ON dcp.id_deposito = d.id_deposito AND dcp.id_producto = p.id_producto
                LEFT JOIN
                    inventario inv ON inv.id_deposito = d.id_deposito AND inv.id_producto = p.id_producto
                LEFT JOIN
                    ventas_historicas vh ON vh.id_producto = p.id_producto AND vh.id_deposito = d.id_deposito
                WHERE
                    p.id_producto = :id_producto
                GROUP BY
                    p.categoria, d.categorias,
                    dcp.capacidad_maxima_unidades, dcp.capacidad_minima_unidades,
                    inv.cantidad
            """)

            result = conn.execute(query, {"id_producto": id_producto, "deposito_id": deposito_id}).fetchone()

            if result:
                features = dict(result._mapping)
                # Manejo de valores None: Asegurar que las capacidades y stocks sean 0 si son nulos
                features['deposito_capacidad_maxima_producto'] = features.get('deposito_capacidad_maxima_producto') or 0
                features['deposito_capacidad_minima_producto'] = features.get('deposito_capacidad_minima_producto') or 0
                features['stock_actual_producto_en_deposito'] = features.get('stock_actual_producto_en_deposito') or 0
                features['total_vendido_producto_deposito_historico'] = features.get('total_vendido_producto_deposito_historico') or 0

                return features
            # Si no se encontró el producto o el depósito para las features específicas (ej. en dcp), retorna None
            return None
    except Exception as e:
        print(f"❌ Error al obtener las características para el producto {id_producto} en depósito {deposito_id}: {e}")
        return None

def get_ingresos_pendientes():
    """
    Obtiene registros de ingreso_producto que no tienen un depósito asignado.
    """
    engine = get_engine()
    if not engine:
        print("❌ No se pudo conectar a la base de datos para obtener ingresos pendientes.")
        return []
    try:
        with engine.connect() as conn:
            query = text("SELECT id_ingreso, id_producto, cantidad FROM ingreso_producto WHERE deposito_asignado IS NULL")
            result = conn.execute(query).fetchall()
            print(f"ℹ️ Encontrados {len(result)} ingresos pendientes de asignación.")
            return [{**row._mapping} for row in result]
    except Exception as e:
        print(f"❌ Error al obtener ingresos de productos pendientes: {e}")
        return []

def get_producto_categoria(id_producto):
    """Obtiene la categoría de un producto a partir de su ID."""
    engine = get_engine()
    if not engine:
        print("❌ No se pudo conectar a la base de datos para obtener la categoría del producto.")
        return None
    try:
        with engine.connect() as conn:
            query = text("SELECT categoria FROM producto WHERE id_producto = :id_producto")
            result = conn.execute(query, {"id_producto": id_producto}).scalar_one_or_none()
            return result
    except Exception as e:
        print(f"❌ Error al obtener la categoría para el producto {id_producto}: {e}")
        return None

def actualizar_ingreso_con_deposito(id_ingreso, id_deposito_sugerido):
    """
    Actualiza la columna 'deposito_asignado' en la tabla 'ingreso_producto'
    para un registro específico.
    """
    engine = get_engine()
    if not engine:
        print("❌ No se pudo conectar a la base de datos para actualizar ingreso_producto.")
        return False
    try:
        with engine.connect() as conn:
            query = text("""
                UPDATE ingreso_producto
                SET deposito_asignado = :deposito_asignado
                WHERE id_ingreso = :id_ingreso
            """)
            # Convertir a int() para evitar el error de tipo NumPy
            conn.execute(query, {"deposito_asignado": int(id_deposito_sugerido), "id_ingreso": id_ingreso})
            conn.commit() # ¡Importante! Confirmar la transacción
            print(f"✅ Registro de ingreso {id_ingreso} actualizado con depósito {int(id_deposito_sugerido)}.")
            return True
    except Exception as e:
        print(f"❌ Error al actualizar el depósito para el ingreso {id_ingreso}: {e}")
        return False

# --- Función Principal de Predicción ---

def predecir_deposito_sugerido(producto_info, model, preprocessor, label_encoder_y):
    """
    Predice el depósito sugerido para un nuevo ingreso de producto,
    considerando reglas de negocio y el modelo de Machine Learning.
    """
    prod_id = producto_info["id_producto"]
    prod_categoria = producto_info["producto_categoria"]
    cantidad_a_ingresar = producto_info["cantidad_a_ingresar"]

    # --- Paso 1: Obtener información de todos los depósitos ---
    todos_los_depositos_info = get_todos_los_depositos_info()
    if not todos_los_depositos_info:
        print("❌ Error: No se pudo obtener información de ningún depósito de la base de datos.")
        return None

    # --- Paso 2: Filtrar depósitos por categorías aceptadas (Regla de Negocio - Pre-ML) ---
    depositos_validos_por_categoria = []

    for dep_info in todos_los_depositos_info:
        dep_id = dep_info["id_deposito"]
        raw_categorias = dep_info.get("categorias", []) # Columna 'categorias' de la tabla 'deposito'

        dep_categorias_aceptadas_list = []
        # Normalizar la entrada para que sea una cadena y luego procesarla
        if raw_categorias:
            # Limpiar caracteres comunes de arrays o JSON ([], "", '')
            cleaned_string = str(raw_categorias).replace('[', '').replace(']', '').replace('"', '').replace("'", "")
            # Dividir por comas y limpiar espacios, convertir a minúsculas
            dep_categorias_aceptadas_list = [c.strip().lower() for c in cleaned_string.split(',') if c.strip()]
        
        # Convierte la categoría del producto a minúsculas para una comparación insensible a mayúsculas/minúsculas
        if prod_categoria.lower() in dep_categorias_aceptadas_list:
            depositos_validos_por_categoria.append(dep_id)
            print(f"✅ Depósito {dep_id} SÍ acepta la categoría '{prod_categoria}'. Considerado candidato.")
        else:
            # print(f"ℹ️ Depósito {dep_id} no acepta la categoría '{prod_categoria}'. Descartado por filtro de negocio.")
            pass # Quité este print para no saturar la salida si hay muchos depósitos descartados

    # --- ALERTA CLAVE: Si ninguna categoría de depósito coincide ---
    if not depositos_validos_por_categoria:
        print(f"🚨 ALERTA: No hay ningún depósito registrado en la base de datos que acepte la categoría '{prod_categoria}'.")
        print("No se puede sugerir un depósito para esta categoría de producto.")
        return None

    # --- Paso 3: Para cada depósito candidato válido, obtener características y preparar para ML ---
    depositos_info_para_ml = []
    for dep_id in depositos_validos_por_categoria:
        deposito_features = get_deposito_candidato_features(prod_id, dep_id)
        if deposito_features:
            # Añadir id_deposito original para referencia posterior, antes del preprocesamiento
            deposito_features['id_deposito_original'] = dep_id
            depositos_info_para_ml.append(deposito_features)
        else:
            print(f"⚠️ No se pudieron obtener datos de capacidad/inventario para producto {prod_id} en depósito {dep_id}. Se omite.")

    if not depositos_info_para_ml:
        print(f"🤔 Después del filtrado por categoría, no se encontraron datos completos de características para ningún depósito candidato.")
        return None

    # Convierte la lista de diccionarios a un DataFrame de Pandas
    df_prediccion = pd.DataFrame(depositos_info_para_ml)

    # Guarda los IDs de depósito originales para remapear después de la predicción
    original_deposito_ids = df_prediccion['id_deposito_original']
    df_prediccion = df_prediccion.drop(columns=['id_deposito_original']) # Eliminar esta columna antes de pasar al preprocesador

    # Preprocesar los datos usando el preprocesador entrenado
    try:
        X_pred = preparar_datos_para_prediccion(df_prediccion, preprocessor)
    except Exception as e:
        print(f"❌ Error durante el preprocesamiento de los datos para predicción: {e}")
        return None

    # Realizar predicciones de probabilidad
    probabilidades = model.predict_proba(X_pred)

    # Mapear las probabilidades a los depósitos originales (descodificados)
    # Esto asegura que las columnas de 'probabilidades' corresponden a los depósitos correctos
    clases_depositos_ml = label_encoder_y.inverse_transform(model.classes_)
    
    # Crear un DataFrame para asociar probabilidades con los IDs de depósito originales
    df_probabilidades = pd.DataFrame(probabilidades, columns=clases_depositos_ml)
    df_probabilidades['id_deposito_original'] = original_deposito_ids.values # Añadir IDs originales

    sugerencias_con_probabilidad = []

    for idx, row_pred_info in df_probabilidades.iterrows():
        dep_id_actual = row_pred_info['id_deposito_original']
        
        # La probabilidad para este depósito (dep_id_actual) de ser la clase correcta
        # Usamos .get() con un valor por defecto de 0.0 si la columna no existe
        probabilidad_para_este_deposito = row_pred_info.get(dep_id_actual, 0.0)

        # Necesitamos la información completa del depósito para las reglas de capacidad
        info_deposito_completa = next((d for d in depositos_info_para_ml if d['id_deposito_original'] == dep_id_actual), None)
        
        if info_deposito_completa:
            sugerencias_con_probabilidad.append({
                'id_deposito': dep_id_actual,
                'probabilidad': probabilidad_para_este_deposito,
                'stock_actual_producto_en_deposito': info_deposito_completa.get('stock_actual_producto_en_deposito', 0),
                'deposito_capacidad_maxima_producto': info_deposito_completa.get('deposito_capacidad_maxima_producto', 0),
                'deposito_capacidad_minima_producto': info_deposito_completa.get('deposito_capacidad_minima_producto', 0)
            })

    # --- Paso 4: Filtrar candidatos por reglas de capacidad (Regla de Negocio - Post-ML) ---
    depositos_filtrados_por_capacidad = []
    umbral_ocupacion_max = 0.95 # 95% de ocupación máxima permitida

    for dep_sug in sugerencias_con_probabilidad:
        dep_id = dep_sug['id_deposito']
        stock_actual = dep_sug['stock_actual_producto_en_deposito']
        cap_max_prod = dep_sug['deposito_capacidad_maxima_producto']
        # cap_min_prod = dep_sug['deposito_capacidad_minima_producto'] # No se usa en esta lógica, pero está disponible

        unidades_despues_ingreso = stock_actual + cantidad_a_ingresar

        motivos_descarte = []

        if cap_max_prod > 0: # Si hay una capacidad máxima definida
            ocupacion_unidades = unidades_despues_ingreso / cap_max_prod
            espacio_libre = cap_max_prod - stock_actual
            
            if ocupacion_unidades > umbral_ocupacion_max:
                motivos_descarte.append(f"Alta ocupación de unidades ({ocupacion_unidades:.2f}) después del ingreso.")
            if espacio_libre < cantidad_a_ingresar:
                motivos_descarte.append(f"Espacio libre insuficiente ({espacio_libre} unidades) para el ingreso de {cantidad_a_ingresar} unidades.")
        elif cantidad_a_ingresar > 0: # Si no hay capacidad máxima definida (o es 0) y hay unidades a ingresar
            motivos_descarte.append(f"Capacidad máxima no definida (es cero), y hay {cantidad_a_ingresar} unidades a ingresar.")
        
        # Si la capacidad máxima es 0 y las unidades a ingresar también son 0, entonces no hay problema.
        # Si la capacidad máxima es 0 y las unidades a ingresar son > 0, es un problema.
        # Por lo tanto, la condición `elif cantidad_a_ingresar > 0` cubre el caso.


        if not motivos_descarte:
            depositos_filtrados_por_capacidad.append(dep_sug)
        else:
            print(f"⚠️ Depósito {dep_id} (Prob: {dep_sug['probabilidad']:.4f}) descartado por:")
            for motivo in motivos_descarte:
                print(f"   - {motivo}")

    # --- Paso 5: Seleccionar el mejor depósito entre los restantes ---
    if not depositos_filtrados_por_capacidad:
        print("🤔 Después de aplicar las reglas de capacidad, no quedan depósitos adecuados.")
        return None
    
    # Elegir el depósito con la probabilidad más alta (entre los que cumplen todas las reglas)
    mejor_deposito = max(depositos_filtrados_por_capacidad, key=lambda x: x['probabilidad'])

    print(f"📦 El modelo sugiere enviar el producto al depósito: {mejor_deposito['id_deposito']} (Probabilidad: {mejor_deposito['probabilidad']:.4f})")
    return mejor_deposito['id_deposito']

# --- Bloque Principal de Ejecución (Solo si el script se ejecuta directamente) ---
if __name__ == "__main__":
    # Cargar modelo y componentes una sola vez
    model, preprocessor, label_encoder_y = load_model_and_components()

    if not all([model, preprocessor, label_encoder_y]):
        print("❌ Error crítico: No se pudieron cargar los componentes del modelo. Abortando.")
        sys.exit(1) # Salir del script si no se puede continuar

    # 1. Obtener todos los ingresos con 'deposito_asignado' en NULL
    ingresos_pendientes = get_ingresos_pendientes()

    if not ingresos_pendientes:
        print("✨ No hay ingresos pendientes para procesar. ¡Todo al día!")
    else:
        # 2. Iterar sobre cada ingreso pendiente y procesarlo
        for ingreso in ingresos_pendientes:
            id_ingreso = ingreso["id_ingreso"]
            id_producto = ingreso["id_producto"]
            cantidad_a_ingresar = ingreso["cantidad"]
            
            print(f"\n--- Procesando Ingreso ID: {id_ingreso}, Producto ID: {id_producto}, Cantidad: {cantidad_a_ingresar} ---")

            # Obtener la categoría del producto, que es necesaria para la predicción
            categoria_producto = get_producto_categoria(id_producto)
            
            if not categoria_producto:
                print(f"❌ No se pudo encontrar la categoría para el producto {id_producto}. Saltando este ingreso.")
                continue

            # Preparar la información para el predictor
            producto_a_predecir = {
                "id_producto": id_producto,
                "producto_categoria": categoria_producto,
                "cantidad_a_ingresar": cantidad_a_ingresar,
            }
            
            # 3. Llamar a la función de predicción
            deposito_sugerido = predecir_deposito_sugerido(
                producto_a_predecir, model, preprocessor, label_encoder_y
            )

            # 4. Si se obtiene una sugerencia, actualizar la base de datos
            if deposito_sugerido is not None: # Usar 'is not None' para manejar 0 o False si fueran posibles sugerencias
                actualizar_ingreso_con_deposito(id_ingreso, deposito_sugerido)
            else:
                print(f"🚫 No se pudo determinar un depósito para el ingreso {id_ingreso}. Permanecerá como pendiente.")