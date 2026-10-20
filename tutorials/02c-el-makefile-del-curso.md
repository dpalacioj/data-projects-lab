# Tutorial 02c — El `Makefile` del curso, explicado línea por línea

> Pre-requisito: haber hecho el [tutorial 00](00-ambientes-venv-conda-uv/) y tener el ambiente del curso montado.

El `Makefile` en la raíz del repositorio es la **interfaz única** para
instalar, testear y limpiar el proyecto. En vez de recordar comandos largos
como `uv sync --extra segmentacion-clientes --extra dev`, escribes
`make setup-caso`.

Este tutorial responde:

- ¿Qué es Make?
- ¿Para qué sirve en un proyecto como este?
- ¿Qué significa cada línea del `Makefile` del repo?
- ¿Qué hago si no quiero (o no puedo) usar Make?

---

## ¿Qué es Make?

`make` es una herramienta de UNIX de 1976 (sí, cincuenta años). Lee un
archivo llamado `Makefile` que declara: "si quieres X, corre estos
comandos". Cada bloque se llama **receta**, tiene un nombre, y la invocas
con `make <nombre>`.

En la práctica funciona como un **atajo con nombres**: en vez de que todos
los estudiantes memoricen `uv sync --extra dev`, defines `make setup` una
sola vez y todos escriben lo mismo.

**Disponibilidad por SO:**

- **macOS y Linux:** `make` ya viene instalado.
- **Windows:** no viene por defecto. Opciones:
  - `winget install GnuWin32.Make`.
  - Copiar a mano los comandos del `Makefile` (cada receta tiene su comando debajo). La última sección de este tutorial trae la tabla de equivalencias.

---

## Mira el `Makefile` real del repo

Abre [`../Makefile`](../Makefile) en tu editor. Lo recorremos de arriba a
abajo.

### 1. El header

```makefile
# Interfaz unica de comandos del repositorio.
# `make` sin argumentos muestra esta ayuda.
```

Comentarios. Make los ignora. Sirven para que el humano sepa qué está
leyendo.

### 2. `.DEFAULT_GOAL := help`

```makefile
.DEFAULT_GOAL := help
```

`.DEFAULT_GOAL` es una **variable especial** de Make. Dice: "si el usuario
escribe `make` sin especificar qué, corre la receta `help`". Sin esto, Make
corre la primera receta del archivo, lo cual es inconsistente y difícil de
recordar.

### 3. `.PHONY: ...`

```makefile
.PHONY: help setup setup-caso smoke test lint clean
```

`.PHONY` le dice a Make: "estos nombres NO son archivos, son solo etiquetas
de recetas". ¿Por qué importa?

Make nació para **compilar** código C. `make hello.o` significaba "produce
el archivo `hello.o`". Si `hello.o` ya existía y era más nuevo que su
código fuente, Make lo saltaba (optimización razonable).

Pero nuestras recetas no producen archivos: `make setup` instala
dependencias. Si alguien crea por casualidad un archivo llamado `setup` en
el repo, Make vería `make setup` y diría "ya existe `setup`, no hago
nada". `.PHONY` lo evita: **forza correr la receta siempre**, exista o no
un archivo con ese nombre.

### 4. La receta `help`

```makefile
help:  ## Lista los comandos disponibles.
	@echo "Comandos disponibles:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo ""
```

Esta receta se ve densa. Vamos pieza por pieza.

**`help:`** — el nombre de la receta.

**`## Lista los comandos disponibles.`** — es un comentario con **dos
almohadillas**. ¿Por qué dos? Es una **convención** (no una regla de Make):
cualquier texto después de `##` en la línea del nombre se usa como
"descripción visible" del comando. Es lo que aparece a la derecha cuando
corres `make help`.

**El `@` al inicio de cada línea** — le dice a Make: "ejecuta este comando
pero **no lo imprimas antes** de ejecutarlo". Sin `@`, Make imprime el
comando y luego su salida, generando ruido. Si quieres ver qué pasa, quita
un `@` y corre `make help`.

**Las líneas con comandos van con `TAB` al inicio**, no con espacios. Es
rigor histórico de Make: si pones espacios en vez de tab, el archivo
falla. Tu editor normalmente lo maneja solo.

**`echo "Comandos disponibles:"`** — comando del shell que imprime texto.
Nada especial de Make.

**`grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST)`** — busca en el
propio Makefile las líneas que lucen como `<nombre>: ## <descripción>`. Es
decir, encuentra las recetas documentadas.

- `grep -E` es una herramienta que busca líneas en archivos con una expresión regular.
- `^[a-zA-Z_-]+:` significa "al inicio de la línea, uno o más caracteres de nombre, luego dos puntos".
- `.*?## .*$` dice "cualquier cosa, luego `## `, luego cualquier cosa hasta el final".
- `$(MAKEFILE_LIST)` es otra variable especial de Make: "el archivo o archivos Makefile que estoy leyendo".
- `$$` es cómo se escribe `$` en un Makefile. Make se come un `$` como propio (lo usa para sus variables), así que para pasarle `$` al shell escribes `$$`.

**`| sort`** — pasa la salida al comando `sort` que ordena las líneas alfabéticamente.

**`| awk '...'`** — pasa el resultado a `awk`, un pequeño lenguaje para procesar texto por líneas. Lo que hace aquí:

- `BEGIN {FS = ":.*?## "}` — antes de procesar, define el separador como `:<lo que sea>## `. Así cada línea queda partida en dos columnas: el nombre a la izquierda, la descripción a la derecha.
- `{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}` — imprime el nombre en cian, luego la descripción en texto normal. `\033[36m` es un código ANSI para color cian; `\033[0m` apaga el color. `$$1` y `$$2` (otra vez el doble `$`) son "la primera y segunda columna" después de partir por `## `.

**El resultado:** una tabla de comandos auto-generada a partir de los `##`
de cada receta. Añades una receta nueva con su `##`, y aparece sola en el
help. **Nunca tocas la receta `help` para mantenerla al día.**

### 5. Las recetas útiles

```makefile
setup:  ## Instala base del curso + grupo dev (equivale a `uv sync --extra dev`).
	uv sync --extra dev
```

Esto ya es fácil: `setup` corre `uv sync --extra dev`. El `## ...` es la
descripción que verás en `make help`.

Las demás recetas (`setup-caso`, `smoke`, `test`, `test-slow`, `lint`,
`clean`) siguen el mismo patrón. Fíjate que no todas usan `@`: en `setup`
Make imprimirá `uv sync --extra dev` antes de ejecutarlo, lo cual es útil
para que el estudiante vea qué se está haciendo. En `help` sí usamos `@`
porque los comandos de formato son ruido.

### 6. `clean`: borra cachés

```makefile
clean:  ## Borra caches de python y de pytest.
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	...
```

- `find .` — busca desde el directorio actual.
- `-type d -name "__pycache__"` — solo directorios con ese nombre.
- `-prune` — "no te metas DENTRO de ese directorio a buscar más". Optimización.
- `-exec rm -rf {} +` — por cada resultado, ejecuta `rm -rf <ruta>`. El `{}` es "el resultado"; `+` significa "agrupa resultados en una sola llamada".

---

## Si no quieres (o no puedes) usar Make

Los comandos del `Makefile` son líneas de shell normales. Si estás en
Windows sin GnuWin32, o simplemente prefieres no instalar Make, abre el
archivo y copia el comando que te interesa:

| En vez de... | Corre a mano |
|---|---|
| `make help` | — (el help existe porque usas Make; no hace falta sin él) |
| `make setup` | `uv sync --extra dev` |
| `make setup-caso` | `uv sync --extra segmentacion-clientes --extra dev` |
| `make smoke` | `python3 scripts/smoke_test.py` |
| `make test` | `uv run pytest` |
| `make test-slow` | `uv run pytest -m slow` |
| `make lint` | `uv run ruff check projects/segmentacion-clientes/` |
| `make clean` | Borra a mano `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/` |

---

## Para profundizar

- Manual oficial de GNU Make: https://www.gnu.org/software/make/manual/
- Un tutorial corto por fuera: https://makefiletutorial.com/

**No necesitas aprender Make a fondo para este curso.** Lo que viste aquí
basta para leer el Makefile del repositorio, modificar una receta
existente, o añadir una nueva.

---

## Qué sigue

Ya puedes leer cualquier Makefile sin asustarte. Si todavía no has montado
el ambiente del curso, vuelve al [tutorial 00](00-ambientes-venv-conda-uv/).
Si ya lo tienes, puedes seguir con la [sesión S02](../sesiones/s02-exploracion-polars/).
