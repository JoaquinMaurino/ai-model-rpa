import os
from datetime import datetime
from sqlalchemy import text
from db.connection import get_engine  # Para obtener la conexión a la DB
from utils.pdfReader import extract_product_info_from_pdf # Para leer los PDFs

# Carpeta donde se encuentran los PDFs de ingresos
CARPETA_PDFS_INGRESOS = "E:\\Development\\ai-model-rpa\\pdfs"

def get_product_id_by_name(product_name, db_engine):
    """
    Busca el ID de un producto en la tabla 'producto' por su nombre.
    """
    if not product_name:
        print("Advertencia: No se proporcionó un nombre de producto para buscar su ID.")
        return None

    if not db_engine:
        print("❌ El motor de la base de datos no está disponible.")
        return None

    try:
        with db_engine.connect() as connection:
            # AJUSTA 'nombre' a la columna real de tu tabla 'producto' que guarda el nombre.
            query = text("SELECT id_producto FROM producto WHERE nombre = :nombre_prod")
            result = connection.execute(query, {"nombre_prod": product_name})
            product_id = result.scalar_one_or_none() # Usa scalar_one_or_none() para obtener un solo valor o None

            if product_id is not None:
                print(f"✅ ID de producto para '{product_name}': {product_id}")
                return product_id
            else:
                print(f"⚠️ Producto '{product_name}' no encontrado en la tabla 'producto'.")
                return None
    except Exception as e:
        print(f"❌ Error al buscar ID de producto para '{product_name}': {e}")
        return None

def insert_ingreso_producto(id_producto, cantidad, fecha_ingreso_str, db_engine):
    """
    Inserta un nuevo registro en la tabla 'ingreso_producto'.
    id_ingreso se genera automáticamente.
    deposito_asignado se inserta como NULL.
    """
    if not all([id_producto, cantidad, fecha_ingreso_str]):
        print("❌ Faltan datos necesarios para insertar en ingreso_producto.")
        return False

    if not db_engine:
        print("❌ El motor de la base de datos no está disponible. No se puede insertar.")
        return False

    try:
        # Convertir la fecha de string (DD-MM-YYYY) a objeto date (YYYY-MM-DD) si tu DB lo requiere así.
        # ReportLab usa DD-MM-YYYY en tu PDF, así que parseamos eso.
        fecha_ingreso_obj = datetime.strptime(fecha_ingreso_str, "%d-%m-%Y").date()

        with db_engine.connect() as connection:
            # La columna id_ingreso se omite porque es autogenerada por la BD.
            # deposito_asignado se inserta explícitamente como NULL.
            query = text("""
                INSERT INTO ingreso_producto (id_producto, cantidad, fecha_ingreso, deposito_asignado)
                VALUES (:id_prod, :cant, :fecha_ing, NULL)
                RETURNING id_ingreso; -- Opcional: para obtener el ID generado
            """)
            
            result = connection.execute(query, {
                "id_prod": id_producto,
                "cant": cantidad,
                "fecha_ing": fecha_ingreso_obj
            })
            
            # Si usas RETURNING id_ingreso
            new_id_ingreso = result.scalar_one()
            connection.commit() # Confirmar la transacción
            
            print(f"✅ Registro insertado en 'ingreso_producto' con ID: {new_id_ingreso}")
            return True

    except Exception as e:
        print(f"❌ Error al insertar en 'ingreso_producto' para producto ID {id_producto}: {e}")
        return False

# --- Lógica principal ---
if __name__ == "__main__":
    print("Iniciando proceso de lectura de PDFs e inserción en la base de datos...")
    
    # Obtener el motor de la base de datos una vez al inicio
    db_engine = get_engine()
    if not db_engine:
        print("No se pudo inicializar la conexión a la base de datos. Abortando.")
        exit()

    processed_pdfs_count = 0
    inserted_records_count = 0

    # Iterar sobre cada PDF en la carpeta de ingresos
    for nombre_archivo_pdf in os.listdir(CARPETA_PDFS_INGRESOS):
        if nombre_archivo_pdf.endswith(".pdf"):
            ruta_completa_pdf = os.path.join(CARPETA_PDFS_INGRESOS, nombre_archivo_pdf)
            print(f"\n--- Procesando PDF: {nombre_archivo_pdf} ---")
            processed_pdfs_count += 1

            # 1. Extraer información del PDF
            info_producto_pdf = extract_product_info_from_pdf(ruta_completa_pdf)

            if info_producto_pdf:
                print(f"  ➡️ Info extraída del PDF: {info_producto_pdf}")
                nombre_producto = info_producto_pdf["producto"]
                cantidad = info_producto_pdf["cantidad"]
                fecha_ingreso_str = info_producto_pdf["fecha_ingreso"]

                # 2. Obtener id_producto de la BD usando el nombre
                id_producto = get_product_id_by_name(nombre_producto, db_engine)

                if id_producto is not None:
                    # 3. Insertar el registro en ingreso_producto
                    print(f"  Intentando insertar ingreso para '{nombre_producto}' (ID: {id_producto})...")
                    if insert_ingreso_producto(id_producto, cantidad, fecha_ingreso_str, db_engine):
                        inserted_records_count += 1
                else:
                    print(f"  ❌ No se pudo obtener el ID de producto para '{nombre_producto}'. No se insertará el ingreso.")
            else:
                print(f"  ❌ No se pudo extraer la información completa del PDF: {nombre_archivo_pdf}.")
    
    print("\n--- Resumen del Proceso ---")
    print(f"Total de PDFs encontrados: {processed_pdfs_count}")
    print(f"Registros de ingreso insertados en DB: {inserted_records_count}")

    if db_engine:
        db_engine.dispose()
        print("Conexiones a la base de datos cerradas.")
        