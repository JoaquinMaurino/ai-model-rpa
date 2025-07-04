from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime, timedelta
import os

# Carpeta donde se guardarán los PDFs
CARPETA = "E:\\Development\\ai-model-rpa\\pdfs"
os.makedirs(CARPETA, exist_ok=True)

# Nombres de productos (deben existir en la base de datos)
nombres_productos_lista = [ # Renombrado para evitar conflicto con la variable 'productos' en el loop
    "Zapatos Deportivos",
    "Set de Bloques"
]

cantidades_lista = [ # Renombrado
    10,
    20
]

# Generar PDFs
for i, nombre_producto in enumerate(nombres_productos_lista): # Iteramos sobre nombres_productos_lista
    cantidad_producto = cantidades_lista[i] # Obtenemos la cantidad correspondiente

    # Nombre del archivo PDF
    nombre_archivo = os.path.join(CARPETA, f"ingreso_{i+1}.pdf")
    c = canvas.Canvas(nombre_archivo, pagesize=A4)
    c.setFont("Helvetica", 12)

    # Dibujamos el Producto y la Cantidad en posiciones separadas
    c.drawString(100, 760, f"Producto: {nombre_producto}")
    c.drawString(100, 740, f"Cantidad: {cantidad_producto}") # Movido a una línea inferior

    # Opcional: Agrega una fecha de ingreso para simular un documento real
    fecha_actual = datetime.now().strftime("%d-%m-%Y")
    c.drawString(100, 720, f"Fecha de Ingreso: {fecha_actual}")

    c.showPage()
    c.save()
    print(f"✅ Archivo creado: {nombre_archivo} con producto '{nombre_producto}'")