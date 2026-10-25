"""Decisiones de negocio del analisis, en UN solo sitio.

Cada constante lleva el porque: el comentario es el contrato con el futuro
mantenedor. Cambiar un numero aqui es un commit revisable; cambiarlo en un
notebook es imposible de rastrear.
"""

# Rango de precios considerados validos. Fuera de aqui son errores de captura
# (precios de 0 euros o productos de 10_000 euros en el catalogo), no ventas.
PRECIO_MIN: float = 0.01
PRECIO_MAX: float = 100.0

# Las facturas que empiezan con esta letra son CANCELACIONES: la Quantity es
# negativa y el registro revierte una venta previa. Para calcular ingresos
# promedio las descartamos.
CANCELACION_PREFIX: str = "C"

# Semilla fija para cualquier muestreo determinista del analisis.
SEMILLA: int = 42
