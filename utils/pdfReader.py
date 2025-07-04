# utils/pdfReader.py
import re
from pypdf import PdfReader
import os # Importar os para la función isfile

def extract_product_info_from_pdf(pdf_path):
    """
    Extrae el nombre del producto, la cantidad y la fecha de ingreso
    de un archivo PDF dado su ruta.

    Args:
        pdf_path (str): La ruta completa al archivo PDF.

    Returns:
        dict: Un diccionario con 'producto', 'cantidad', 'fecha_ingreso' si se encuentran,
              o None si no se puede extraer la información o el archivo no existe.
    """
    if not os.path.isfile(pdf_path):
        print(f"❌ Error: El archivo PDF no existe en la ruta: {pdf_path}")
        return None

    try:
        reader = PdfReader(pdf_path)
        # Asumimos que la información relevante está en la primera página
        page = reader.pages[0]
        text = page.extract_text()

        producto = None
        cantidad = None
        fecha_ingreso = None

        # Expresiones regulares para extraer cada pieza de información
        # Producto: busca "Producto: " y captura lo que sigue hasta el final de la línea
        match_producto = re.search(r"Producto:\s*(.+)", text)
        if match_producto:
            producto = match_producto.group(1).strip()

        # Cantidad: busca "Cantidad: " y captura dígitos
        match_cantidad = re.search(r"Cantidad:\s*(\d+)", text)
        if match_cantidad:
            cantidad = int(match_cantidad.group(1).strip()) # Convertir a entero

        # Fecha de Ingreso: busca "Fecha de Ingreso: " y captura la fecha (DD-MM-YYYY)
        match_fecha = re.search(r"Fecha de Ingreso:\s*(\d{2}-\d{2}-\d{4})", text)
        if match_fecha:
            fecha_ingreso = match_fecha.group(1).strip() # Puedes convertir a datetime si lo necesitas

        if producto and cantidad and fecha_ingreso:
            return {
                "producto": producto,
                "cantidad": cantidad,
                "fecha_ingreso": fecha_ingreso
            }
        else:
            print(f"⚠️ No se pudo extraer toda la información necesaria del PDF: {pdf_path}")
            print(f"  Producto: {producto}, Cantidad: {cantidad}, Fecha: {fecha_ingreso}")
            return None

    except Exception as e:
        print(f"❌ Error al leer o extraer texto del PDF {pdf_path}: {e}")
        return None