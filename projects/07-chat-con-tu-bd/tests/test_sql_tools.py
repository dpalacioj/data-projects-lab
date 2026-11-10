"""Tests de la capa SQL: seguridad y schema inspector."""
import pytest
from chat_bd.sql_tools import (
    SQLNoPermitidoError,
    ejecutar_sql_lectura,
    resumen_bd,
    schema_como_texto,
    validar_sql_lectura,
)


def test_select_simple_pasa_validacion():
    sql = "SELECT * FROM productos"
    assert "SELECT" in validar_sql_lectura(sql).upper()


def test_with_cte_pasa_validacion():
    sql = "WITH t AS (SELECT * FROM productos) SELECT * FROM t"
    validar_sql_lectura(sql)


def test_insert_es_rechazado():
    with pytest.raises(SQLNoPermitidoError):
        validar_sql_lectura("INSERT INTO productos VALUES (99, 'x', 1)")


def test_update_es_rechazado():
    with pytest.raises(SQLNoPermitidoError):
        validar_sql_lectura("UPDATE productos SET precio = 0")


def test_delete_es_rechazado():
    with pytest.raises(SQLNoPermitidoError):
        validar_sql_lectura("DELETE FROM productos")


def test_drop_es_rechazado():
    with pytest.raises(SQLNoPermitidoError):
        validar_sql_lectura("DROP TABLE productos")


def test_multi_sentencia_es_rechazada():
    with pytest.raises(SQLNoPermitidoError):
        validar_sql_lectura("SELECT 1; SELECT 2")


def test_select_con_delete_en_string_literal_tambien_cae():
    """Falso positivo conservador: preferimos rechazar a dejar pasar."""
    with pytest.raises(SQLNoPermitidoError):
        validar_sql_lectura("SELECT 'DELETE' FROM productos")


def test_ejecutar_devuelve_dataframe():
    df = ejecutar_sql_lectura("SELECT * FROM productos")
    assert len(df) == 5
    assert "nombre" in df.columns


def test_ejecutar_inyecta_limit_si_no_hay():
    """Si el SELECT no tiene LIMIT, se le mete uno para proteger al LLM."""
    df = ejecutar_sql_lectura("SELECT * FROM productos", limite=2)
    assert len(df) == 2


def test_ejecutar_respeta_limit_del_usuario():
    df = ejecutar_sql_lectura("SELECT * FROM productos LIMIT 3", limite=1000)
    assert len(df) == 3


def test_resumen_bd_devuelve_tablas_y_muestras():
    r = resumen_bd()
    assert set(r["tablas"].keys()) == {"productos", "ventas"}
    assert r["tablas"]["productos"]["n_filas"] == 5
    assert len(r["tablas"]["productos"]["muestra"]) == 3


def test_schema_como_texto_incluye_todas_las_tablas():
    texto = schema_como_texto()
    assert "productos" in texto
    assert "ventas" in texto
    assert "5 filas" in texto or "5," in texto
