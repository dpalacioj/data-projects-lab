# Tutorial 01c · Tu página web con GitHub Pages (sin crear contenido nuevo)

> Pre-requisito: haber hecho el tutorial [01 (git)](01-git-flujo-de-trabajo.md).
> Tener un repositorio tuyo en GitHub.

Vas a aprender a convertir tu repositorio en un **sitio web público**
usando GitHub Pages, **sin escribir contenido nuevo**. Los mismos
`README.md` y otros archivos `.md` que ya tienes serán renderizados
como páginas HTML.

Al terminar tendrás una URL tipo:

```
https://TU-USUARIO.github.io/TU-REPO/
```

Dura ~15 minutos.

---

## Parte 1 — Teoría corta

### Qué es GitHub Pages

Es un servicio gratuito de GitHub que convierte archivos de tu repo
en un sitio web. Usa [Jekyll](https://jekyllrb.com/) por debajo: un
generador de sitios estáticos que lee tus `.md`, les aplica un tema
HTML/CSS, y publica el resultado.

Caso típico: todos los README de tu repo se vuelven páginas navegables
con tipografía y estilos; los links entre ellos siguen funcionando.

### "¿Me quedo con dos ramas?"

**No necesariamente.** GitHub Pages te ofrece tres fuentes posibles:

| Opción | Qué significa | Ramas que tienes al final |
|---|---|---|
| **main /** (raíz) | Sirve desde la raíz de `main` | Solo `main` |
| **main /docs** | Sirve desde la subcarpeta `docs/` de `main` | Solo `main` |
| **`gh-pages` (rama separada)** | Sirve desde otra rama dedicada | `main` + `gh-pages` |

En este tutorial usamos **main /** porque:

- Es la más simple (no mantienes otra rama en paralelo).
- Cada vez que empujes a `main`, el sitio se actualiza automático en ~1 minuto.
- Tus `README.md` ya están donde Jekyll los espera.

**Decisión:** una sola rama, `main`. Nada de `gh-pages`. Si en el
futuro quieres separar sitio y código, puedes mover a `gh-pages`
después.

### Qué URL vas a tener

Si tu repo es `mi-usuario/mi-repo`, la URL será:

```
https://mi-usuario.github.io/mi-repo/
```

(Caso especial: si el repo se llama **exactamente** `mi-usuario.github.io`,
la URL es `https://mi-usuario.github.io/` sin subpath. Eso es para
landing personales, no es nuestro caso aquí.)

---

## Parte 2 — El único archivo que añades: `_config.yml`

Para que Jekyll sepa qué incluir, cómo verse y qué ignorar, creas
**un único archivo nuevo** en la raíz de tu repo: `_config.yml`.

### Versión mínima (3 líneas)

```yaml
title: Mi Proyecto
description: Breve descripción
theme: jekyll-theme-cayman
```

Con esto, Jekyll ya hace lo siguiente:

- Renderiza tu `README.md` raíz como **la página de inicio**.
- Renderiza cada otro `.md` como página hija.
- Aplica el tema visual que elegiste.

**Pero** procesa TODO el repo por defecto: tus `.ipynb`, tu `src/`, tus
datos, tus modelos. Eso es ruido. Por eso añadimos un `exclude:`.

### Versión recomendada para un proyecto de ciencia de datos

```yaml
title: Mi Proyecto
description: Breve descripción
theme: jekyll-theme-cayman

exclude:
  # Notebooks y codigo
  - "**/*.ipynb"
  - "**/*.py"
  - "**/__pycache__/"

  # Datos y modelos
  - "**/*.csv"
  - "**/*.joblib"
  - "**/*.pth"
  - "**/*.parquet"

  # Subcarpetas tipicas
  - "**/src/"
  - "**/app/"
  - "**/tests/"
  - "**/data/"
  - "**/models/"
  - "**/notebooks/"

  # Configs del proyecto
  - uv.lock
  - pyproject.toml
  - Makefile
```

### Línea por línea: qué significa cada clave

| Clave | Qué hace |
|---|---|
| `title` | Título que aparece en la pestaña del navegador y en el header del sitio |
| `description` | Subtítulo que aparece bajo el header (depende del tema) |
| `theme` | Estilo visual. GitHub ofrece [~10 temas gratuitos](https://pages.github.com/themes/). "Cayman" es limpio y popular |
| `exclude` | Lista de archivos/carpetas que Jekyll NO procesa. Usa patrones *glob*: `**/` significa "cualquier profundidad", `*.py` significa "cualquier archivo .py" |

**Temas alternativos** (reemplaza en `theme:` y empuja; se actualiza en 1 min):

- `jekyll-theme-cayman` — limpio, header con gradiente azul.
- `jekyll-theme-minimal` — minimalista, blanco y negro.
- `jekyll-theme-slate` — oscuro.
- `jekyll-theme-architect` — tipografía marcada.
- `jekyll-theme-tactile` — compacto.

### Línea por línea: qué significa cada `exclude`

| Patrón | Qué omite |
|---|---|
| `"**/*.ipynb"` | Todos los notebooks de Jupyter, en cualquier carpeta |
| `"**/*.py"` | Todo el código Python |
| `"**/__pycache__/"` | Caches de Python compiladas |
| `"**/*.csv"` | Datos tabulares. Si tienes CSVs pedagógicos pequeños que SÍ quieras servir, bórralos de este patrón |
| `"**/*.joblib"`, `"**/*.pth"` | Modelos entrenados serializados |
| `"**/src/"`, `"**/app/"`, `"**/tests/"` | Carpetas típicas con código |
| `"**/data/"`, `"**/models/"`, `"**/notebooks/"` | Carpetas con datos o modelos |
| `uv.lock`, `pyproject.toml`, `Makefile` | Configuración interna del proyecto |

**Patrón importante:** las carpetas DEBEN terminar con `/` para que
Jekyll las trate como carpetas y no como archivos.

---

## Parte 3 — Pasos en GitHub (lo que haces tú)

### Paso 1 — Añade `_config.yml` a tu repo

Desde tu máquina, siguiendo el flujo del tutorial 01 (nunca en main):

```bash
git switch -c feat/github-pages
# crea el archivo con tu editor preferido
# (nano, vim, VS Code, TextEdit, lo que uses)
```

Pega el contenido del `_config.yml` recomendado (de la Parte 2).

```bash
git add _config.yml
git commit -m "feat: anadir configuracion de GitHub Pages"
git push -u origin feat/github-pages
```

Abre el Pull Request en GitHub y mergealo a `main` (ver el
[tutorial 01](01-git-flujo-de-trabajo.md#paso-5--abrir-el-pull-request)).

### Paso 2 — Activa Pages en Settings

En tu repo en github.com:

1. Click en **Settings** (pestaña superior derecha).
2. En el menú lateral izquierdo, click en **Pages**.
3. En **"Source"** selecciona: **"Deploy from a branch"**.
4. En **"Branch"**:
   - **Branch**: `main`
   - **Folder**: `/ (root)`
5. Click **Save**.

### Paso 3 — Espera 1 minuto

GitHub arranca a construir tu sitio. En la pestaña **Actions** del
repo verás un workflow llamado `pages-build-deployment` corriendo.

Cuando termine (dura entre 30 segundos y 1 minuto la primera vez),
vuelve a **Settings → Pages**. Verás arriba:

```
Your site is live at https://TU-USUARIO.github.io/TU-REPO/
```

### Paso 4 — Visita la URL

Abre esa URL. Deberías ver tu `README.md` renderizado como página web
con el tema que elegiste. Click en los links internos: te llevan a
las otras páginas del sitio.

---

## Parte 4 — ¿Cómo se actualiza?

Cada vez que haces push a `main`, GitHub reconstruye el sitio
automático en ~1 minuto. No tienes que hacer nada más.

Si editaste un `.md`, hiciste push, y en 2 minutos no ves el cambio:

- Ve a **Actions** del repo.
- Mira el último workflow `pages-build-deployment`.
- Si tiene check rojo ❌, abre el log — generalmente es un error de sintaxis en el `_config.yml` o un markdown mal formado.

---

## Parte 5 — Cambios de tema y mantenimiento

### Cambiar el tema

Edita `_config.yml`, cambia `theme:` por uno de los
[temas oficiales de GitHub Pages](https://pages.github.com/themes/).
Push a main. 1 minuto después el sitio tiene el nuevo tema.

### Añadir una página nueva

Simplemente crea un nuevo `.md` en el repo. Jekyll lo detecta y lo
convierte en una página del sitio. La URL será la ruta del archivo
(p. ej. `docs/mi-pagina.md` → `https://TU-USUARIO.github.io/TU-REPO/docs/mi-pagina.html`).

### Dejar un archivo fuera del sitio pero sí en el repo

Añádelo al `exclude:` del `_config.yml`.

### Dar de baja el sitio

Settings → Pages → cambia **Source** a *"None"* y guarda. El sitio
desaparece, pero el repo queda intacto.

---

## Problemas comunes

### "404 page not found"

- Verifica que `_config.yml` está en la **raíz** del repo, no en una subcarpeta.
- Verifica que la URL tiene el nombre del repo correcto (con guiones, en minúsculas).
- La primera vez puede tardar hasta 10 minutos. Si llevas más, mira Actions.

### Mi página se ve rota (sin estilos)

- Verifica que el nombre del `theme` es correcto. Los nombres válidos están en https://pages.github.com/themes/.
- Chequea en Actions si hubo un error de build.

### Mis notebooks no se renderizan como páginas

- Correcto, están excluidos a propósito. Los `.ipynb` no son HTML limpio.
- Si quieres compartir un notebook, enlaza a [nbviewer](https://nbviewer.org/) con la URL del notebook en github.com, o a github.com directo (que los renderiza decente).

### Un `.md` importante no aparece en el sitio

- Chequea que no esté en una carpeta excluida en el `_config.yml`.
- Chequea que el nombre del archivo NO empiece con `_` (Jekyll ignora cosas que empiezan con `_` por convención).

### Error "Invalid YAML" en Actions

- Tu `_config.yml` tiene mala indentación o un carácter raro.
- La indentación de YAML es con **espacios, no tabs**. Dos espacios por nivel.
- Las comillas en strings tipo `"**/*.py"` deben ser consistentes (todas dobles o todas simples).

---

## Resumen

- **Una** rama (`main`), **un** archivo nuevo (`_config.yml`).
- Lo que escribes en los `.md` del repo **es** el contenido del sitio.
- Activas Pages una vez en Settings; después todo es automático.
- Push a main → sitio actualizado en 1 minuto.

Si quieres ver el ejemplo real, el `_config.yml` de este mismo repo
(`data-projects-lab`) usa exactamente este patrón y publica todos los
tutoriales, sesiones y READMEs de proyecto como sitio navegable.
