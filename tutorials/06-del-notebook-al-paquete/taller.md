# Taller 02e — Convierte tu notebook en paquete

> Se puede terminar en clase o quedar como tarea.
> **Entregable:** un repositorio (o una carpeta en el tuyo) con la estructura
> mínima del estado 3, un notebook que importa del paquete, y al menos 2 tests
> verdes.

---

## 1. Elige un notebook (10 min)

Toma un notebook tuyo (de este curso o de otro). Si no tienes ninguno a mano,
usa el `01-tres-estados.ipynb` de este tutorial como insumo.

**Criterio de elección:** debe tener al menos:

- Una lectura de datos.
- Al menos **dos reglas de limpieza o transformación** (filtros, descartes,
  conversiones).
- Al menos **una constante de negocio** (un umbral, un rango, un prefijo).
- Un cálculo final (una métrica, una tabla, un gráfico).

Si tu notebook es solo "cargar y mostrar", el ejercicio no vale la pena.

---

## 2. Esqueleto del paquete (10 min)

Crea esta estructura mínima en tu carpeta:

```
mi-analisis/
├── pyproject.toml
├── src/
│   └── mi_analisis/
│       ├── __init__.py
│       ├── config.py
│       ├── cargar.py
│       ├── limpiar.py
│       └── metricas.py
├── tests/
│   └── test_limpiar.py
└── notebook.ipynb
```

Usa el [`paquete/`](paquete/) de este tutorial como referencia. El
`pyproject.toml` mínimo cabe en 10 líneas.

---

## 3. Mueve las constantes a `config.py` (10 min)

Toma TODAS las constantes de negocio de tu notebook (umbrales, rangos,
prefijos, semillas) y colócalas en `config.py` con **un comentario por
constante** diciendo qué significan y por qué ese valor.

**Qué cuenta como constante de negocio:**
- `PRECIO_MIN = 0.01` → sí.
- `np.pi` → no (es matemática, no negocio).
- `random_state=42` → sí (es una decisión reproducible).

**Qué no cuenta:**
- `x = 5` sin contexto → no es constante de negocio, es ruido.

---

## 4. Divide la lógica en funciones puras (20 min)

Cada regla de limpieza va a su propia función:

```python
# limpiar.py
def descartar_cancelaciones(df): ...
def filtrar_precio_valido(df): ...
def limpiar(df):
    return df.pipe(descartar_cancelaciones).pipe(filtrar_precio_valido)
```

Cada función es:
- **Pura:** entra DataFrame, sale DataFrame. No modifica globales.
- **Nombrada por lo que hace,** no por lo que es ("filtrar_precio_valido" sí,
  "paso_2" no).
- **Importa las constantes desde `config.py`,** no las escribe inline.

El cálculo final (precio medio, exactitud, etc.) va a `metricas.py`.

---

## 5. Escribe al menos 2 tests (15 min)

En `tests/test_limpiar.py`, al menos:

- Un test que verifica una regla específica con un DataFrame construido a
  mano (3-5 filas, nombradas, con casos buenos y malos).
- Un test que verifica el resultado final sobre un dataset mínimo
  (determinista).

Los tests tienen que correr con `pytest tests/` desde la raíz de tu paquete,
sin requerir el dataset completo.

---

## 6. Reescribe el notebook (10 min)

En `notebook.ipynb` ya no tienes constantes, ni funciones, ni lógica. Solo:

```python
from mi_analisis import limpiar, mi_metrica
from mi_analisis.cargar import cargar_datos

df = cargar_datos()
limpio = limpiar(df)
resultado = mi_metrica(limpio)
print(resultado)
```

El notebook se convirtió en **un orquestador delgado**. Toda la lógica es
importada; nada vive en celdas.

---

## 7. Criterios de aceptación

- [ ] Instalable con `uv pip install -e .` (o `pip install -e .`).
- [ ] Todas las constantes de negocio están en `config.py` con comentario.
- [ ] Las reglas de limpieza son funciones puras en `limpiar.py`.
- [ ] Los tests de `tests/` pasan con `pytest`.
- [ ] El notebook importa del paquete y no tiene lógica propia.
- [ ] El resultado del notebook coincide con el resultado del notebook
  original (si uno de los dos difiere, hay un bug en el original; investiga
  antes de aceptar).

Si llegas hasta aquí, cualquier otra persona puede clonar tu repositorio,
correr `uv pip install -e .`, ejecutar tu notebook y obtener el mismo número.
Eso es lo que **"reproducible"** significa cuando se usa en serio.
