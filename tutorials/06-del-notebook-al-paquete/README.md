# Tutorial 02e — Del notebook al paquete: el mismo analisis, tres veces

> Pre-requisito: ambiente del curso montado ([tutorial 00](../00-ambientes-venv-conda-uv/)).
> Dependencia extra: ninguna; usa pandas y numpy que ya vienen en la base.

Un notebook con estado oculto no es reproducible, aunque dé el número correcto
hoy. Este tutorial muestra el mismo cálculo trivial —el precio promedio
ponderado de una tienda ficticia— en **tres estados** del mismo código, y
compara qué cambia:

| Estado | Cómo se ejecuta | Qué falla |
|---|---|---|
| **Caótico** | celdas fuera de orden, `!pip install` a mitad, constantes reescritas a mano | el resultado depende del orden de Shift+Enter |
| **Script** | `python analisis.py`, de arriba abajo | reproducible dentro del archivo, pero no se puede importar ni testear por partes |
| **Paquete** | `from analisis_ventas import ...` | importable, testeable, versionado, con dependencias declaradas |

**El número tiene que salir igual en los tres.** Ese es el punto: la
reproducibilidad no es una propiedad del **resultado**, es una propiedad del
**proceso**. Un notebook caótico que da el número correcto por casualidad y un
paquete que da el mismo número son cosas distintas.

Este tutorial adapta la [sesión s01 de MLOps-Course](https://github.com/dpalacioj/MLOps-Course/tree/main/sesiones/s01-reproducibilidad)
al scope de este curso (pre-MLOps).

---

## Qué hay en esta carpeta

| Archivo / carpeta | Qué contiene |
|---|---|
| [`01-tres-estados.ipynb`](01-tres-estados.ipynb) | El notebook principal con la demostración ejecutable completa. Es lo primero que abres |
| [`datos_sinteticos.py`](datos_sinteticos.py) | Generador determinista del dataset. No requiere descarga externa; produce 5000 filas de ventas con cancelaciones y precios fuera de rango para que las reglas de limpieza tengan algo real que hacer |
| [`paquete/`](paquete/) | El estado 3 implementado: paquete Python mínimo instalable (`analisis-ventas`), con `src/`, `tests/`, `pyproject.toml` |
| [`taller.md`](taller.md) | Aplicación individual: convertir tu propio notebook en un paquete siguiendo el patrón |

---

## Cómo recorrerlo

1. Abre [`01-tres-estados.ipynb`](01-tres-estados.ipynb). Ya trae outputs
   commiteados; léelo primero sin ejecutar para ver qué pasa.
2. Ejecútalo tú mismo. En el estado 3, el notebook instala el paquete local
   con `pip install -e paquete/`. Si te molesta modificar tu entorno, hazlo en
   un `.venv` del tutorial.
3. Abre los archivos del paquete y léelos. Son pequeños a propósito: la
   totalidad del "código de negocio" cabe en menos de 100 líneas.
4. Corre los tests: `pytest paquete/tests/` (desde esta carpeta).
5. Haz el [taller](taller.md): aplica el patrón a un notebook tuyo.

---

## Qué NO cubre este tutorial

- **No** cubre `uv`, `Makefile` ni gestión de ambientes. Eso está en los
  tutoriales [00](../00-ambientes-venv-conda-uv/) y [02c](../02c-el-makefile-del-curso.md).
- **No** cubre contratos de datos ni dataset cards. Eso es el
  [tutorial 02d](../02d-datos-como-codigo/).
- **No** cubre CI, pre-commit hooks ni devcontainer. Esas son piezas de MLOps
  que no están en este curso.

El objetivo acotado: dejar claro **por qué** convertir un notebook en paquete
no es cosmético, y mostrar **cómo** se ve el resultado final.

---

## Qué sigue

Si terminaste el taller, el patrón "paquete + notebook que importa" ya es tuyo.
Un buen siguiente paso es abrir [`projects/00-segmentacion-clientes/`](../../projects/00-segmentacion-clientes/)
y ver cómo se aplica el mismo patrón sobre un dataset real (Online Retail II)
en el caso guía del módulo.
