from sqlalchemy import text
from db.connection import get_engine

def get_dataset_entrenamiento():
    engine = get_engine()
    if engine:
        with engine.connect() as conn:
            query = text("""
                SELECT
                    -- Características del Producto:
                    p.id_producto,
                    p.categoria AS producto_categoria,

                    -- Características del Depósito asignado (la 'verdad' histórica):
                    ip.deposito_asignado AS id_deposito_asignado, -- Esta es tu variable objetivo (target)
                    d.categorias AS deposito_categorias_aceptadas,
                    
                    -- ¡Nuevo! Capacidades específicas del depósito PARA ESE PRODUCTO (de Deposito_Capacidad_Producto)
                    dcp.capacidad_maxima_unidades AS deposito_capacidad_maxima_producto,
                    dcp.capacidad_minima_unidades AS deposito_capacidad_minima_producto,

                    -- ¡Nuevo! Stock actual de ese producto en ese depósito (de Inventario)
                    COALESCE(inv.cantidad, 0) AS stock_actual_producto_en_deposito,

                    -- Total vendido de este producto específico desde ESTE depósito específico en el historial
                    COALESCE(SUM(vh.cantidad), 0) AS total_vendido_producto_deposito_historico
                FROM
                    ingreso_producto ip
                JOIN
                    producto p ON ip.id_producto = p.id_producto
                JOIN
                    deposito d ON ip.deposito_asignado = d.id_deposito
                LEFT JOIN
                    deposito_capacidad_producto dcp ON dcp.id_deposito = d.id_deposito AND dcp.id_producto = p.id_producto
                LEFT JOIN
                    inventario inv ON inv.id_deposito = d.id_deposito AND inv.id_producto = p.id_producto
                LEFT JOIN
                    ventas_historicas vh ON vh.id_producto = p.id_producto AND vh.id_deposito = d.id_deposito
                
                GROUP BY
                    p.id_producto, p.categoria,
                    ip.deposito_asignado, d.categorias, 
                    dcp.capacidad_maxima_unidades, dcp.capacidad_minima_unidades,
                    inv.cantidad -- Para el stock de ESE producto en ESE depósito
                ORDER BY
                    p.id_producto, ip.deposito_asignado
            """)
            result = conn.execute(query)
            return result.fetchall()
    else:
        print("❌ No se pudo conectar a la base de datos.")
        return []