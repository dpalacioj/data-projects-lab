---
title: "01 · Git y GitHub"
parent: Tutoriales
nav_order: 1
---
# Tutorial — Git y GitHub: el flujo de trabajo del curso

Última revisión: 2026-10-25. Prerrequisito: una cuenta de GitHub creada.
Nada más.

Este tutorial **no** es una referencia completa de git. Es la ruta mínima para
que puedas:

1. Clonar este repositorio.
2. Trabajar en una rama propia sin romper el trabajo de nadie.
3. Hacer tu primer **Pull Request** (PR) y entender qué pasa cuando lo abres.

Si nunca has usado git, empieza aquí. Si ya has hecho `git clone` alguna vez
pero nunca un PR, también. Si ya trabajas con PRs todos los días, puedes saltar
al [cheatsheet de comandos](#comandos-que-vas-a-repetir-mil-veces).

---

## ¿Por qué git y por qué este flujo?

**Git** es un sistema de control de versiones: guarda la historia completa de
tu código. Si borras un archivo por error, está. Si cambias algo y rompe
todo, puedes volver atrás. Si dos personas tocan el mismo archivo, git te
avisa en vez de pisar silenciosamente el trabajo de la otra.

**GitHub** es un servicio para alojar repositorios de git en la nube,
colaborar y revisar código.

**El flujo "clone → branch → commit → push → PR"** no es una ceremonia. Es
cómo colabora el mundo real:

- Trabajar directo en `main` es como editar el documento final de la empresa
  en vivo mientras tu jefe lo lee: cualquier error queda publicado al instante.
- Trabajar en una **rama** es como trabajar en una copia privada. Haces cambios,
  los pruebas, y solo cuando están listos propones mezclarlos a `main`.
- El **Pull Request** es la propuesta: "oye, revisen esto antes de que entre a `main`".
- La revisión (humana o automática) atrapa errores antes de que lleguen al código
  que todos usan.

Este flujo es el estándar en empresas, en proyectos open source y en este curso.

---

## Paso 0 — Instalar y configurar git (una sola vez)

### Instalar

| SO | Cómo verificar | Si no está |
|---|---|---|
| macOS | `git --version` | `xcode-select --install` (viene con las herramientas de desarrollo) |
| Linux | `git --version` | `sudo apt install git` (Debian/Ubuntu) o equivalente |
| Windows | `git --version` en PowerShell | Descargar [Git for Windows](https://git-scm.com/download/win) |

### Configurar tu identidad

Git firma cada commit con tu nombre y correo. Esto se hace **una sola vez** en
tu máquina:

```bash
git config --global user.name "Tu Nombre"
git config --global user.email "tu-correo@dominio.com"
```

Usa el **mismo correo** que tienes en GitHub. Si no coincide, los commits
aparecerán en el histórico pero no estarán ligados a tu perfil de GitHub.

### Autenticación con GitHub

GitHub ya no acepta contraseña por línea de comandos. Dos opciones:

- **HTTPS + Personal Access Token (PAT)** — más fácil para empezar. Cuando
  hagas `git push` por primera vez, pedirá usuario y "contraseña". En
  "contraseña" pones un token creado en
  `GitHub → Settings → Developer settings → Personal access tokens`.
- **SSH** — más cómodo a largo plazo. Guía oficial:
  [GitHub Docs - Connecting to GitHub with SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh).

Para el curso, HTTPS + PAT es suficiente.

---

## Paso 1 — Clonar el repositorio

**Clonar** significa descargar una copia completa del repositorio en tu
máquina, con toda la historia.

```bash
git clone https://github.com/dpalacioj/data-projects-lab.git
cd data-projects-lab
```

Después de `clone` tienes una carpeta `data-projects-lab/`. Dentro de ella,
estás por defecto en la rama **`main`** (la "oficial").

Verifica:

```bash
git status
# On branch main
# Your branch is up to date with 'origin/main'.
# nothing to commit, working tree clean
```

Esa salida significa: *estás en `main`, no hay cambios locales, estás
sincronizado con GitHub (`origin`).* Es el estado "limpio" al que vas a querer
volver siempre.

---

## Paso 2 — Crear una rama para tu trabajo

**Regla del curso: nunca trabajes directo en `main`.** Siempre crea una rama.

```bash
git switch -c feat/mi-primer-cambio
```

Explicación:

- `git switch -c <nombre>` crea una rama nueva y te cambia a ella.
- `feat/` es una convención: indica que es una nueva funcionalidad. Otras
  convenciones comunes:
  - `fix/arreglar-grafico` → arreglas un bug.
  - `docs/README-ejemplos` → cambios de documentación.
  - `refactor/limpiar-utils` → reorganización sin cambiar comportamiento.
- El nombre después del `/` debe ser corto y descriptivo. Sin tildes, sin
  espacios (usa guiones).

Verifica dónde estás:

```bash
git branch
#   main
# * feat/mi-primer-cambio
```

El `*` marca tu rama actual.

---

## Paso 3 — Hacer cambios y commitearlos

Edita archivos como lo harías normalmente. Cuando hayas terminado un cambio
coherente, lo "fotografías" con un **commit**.

### El ciclo mínimo

```bash
# 1. Ver qué cambiaste
git status           # qué archivos tocaste
git diff             # qué líneas específicamente cambiaron

# 2. Elegir qué va en este commit
git add archivo.py                 # un archivo específico
git add carpeta/                   # una carpeta
# evita 'git add .' sin leer status primero: puedes commitear basura
# (archivos temporales, credenciales, datos crudos)

# 3. Confirmar el commit con un mensaje
git commit -m "feat: añade cálculo de recencia a RFM"
```

### Cómo escribir mensajes de commit

Este curso usa **Conventional Commits**. Formato:

```
<tipo>: <descripción corta en imperativo>
```

Tipos más comunes: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.

Ejemplos:

- `feat: añade pipeline de limpieza de descripciones` ✅
- `fix: corrige fuga temporal en construcción de features` ✅
- `docs: aclara convención de nombres de ramas` ✅
- `cambios` ❌ (no dice nada)
- `WIP` ❌ (sí, lo hemos visto)
- `arreglé lo del profe` ❌

El mensaje importa porque es lo que verán cuando revisen tu PR y lo que verás
tú misma en seis meses cuando quieras entender qué hiciste.

### ¿Cada cuánto hacer commit?

Cuando terminas **una idea coherente**. No por minutos ni por archivo. Un
commit debe ser algo que, si lo describes en una línea, tenga sentido.

- "Agregué la función de limpiar texto y actualicé el notebook que la usa" → un commit.
- "Agregué cinco funciones distintas, actualicé el README, subí datos y arreglé un bug" → cinco commits.

---

## Paso 4 — Subir tu rama a GitHub

Hasta ahora tu rama solo existe en tu máquina. Para que GitHub la vea:

```bash
git push -u origin feat/mi-primer-cambio
```

Qué significa:

- `push` → sube los commits locales al repositorio remoto.
- `origin` → el apodo del repositorio remoto (lo creó `git clone`).
- `-u` → "esta rama local sigue a esta rama remota de ahora en adelante". Solo
  se usa la primera vez. Después basta con `git push`.

Si es tu primera vez pedirá autenticación (ver [Paso 0](#paso-0--instalar-y-configurar-git-una-sola-vez)).

Al terminar, GitHub te imprime algo como:

```
remote: Create a pull request for 'feat/mi-primer-cambio' on GitHub by visiting:
remote:   https://github.com/dpalacioj/data-projects-lab/pull/new/feat/mi-primer-cambio
```

**Haz click en esa URL.** Te lleva directo al formulario de PR.

---

## Paso 5 — Abrir el Pull Request

En la página de GitHub:

1. Verifica arriba que la "base branch" sea `main` y la "compare branch" sea tu rama.
2. Escribe un **título** corto (misma convención de los commits).
   Ejemplo: `feat: añade cálculo de recencia a RFM`.
3. Escribe una **descripción** respondiendo a tres preguntas:
   - **¿Qué cambia?** (una lista de 2-4 bullets).
   - **¿Por qué?** (qué problema resuelve o qué habilita).
   - **¿Cómo probarlo?** (comando o pasos para que el revisor verifique).
4. Click en **Create pull request**.

Ejemplo de descripción buena:

```markdown
## Qué cambia
- Añade `calcular_recencia()` en `src/segmentacion/features.py`.
- Añade test de contrato en `tests/test_features.py`.

## Por qué
El modelo de recompra necesita esta columna; sin ella el pipeline de la
sesión S03 no corre.

## Cómo probarlo
```bash
make test
```
Debería pasar `test_calcular_recencia`.
```

---

## Paso 6 — ¿Qué pasa cuando abres el PR?

Dos cosas, en paralelo:

### 1. Revisión humana

Alguien (en este curso: el instructor o un compañero) lee los cambios en la
pestaña **Files changed** y deja comentarios. Puede:

- **Aprobar** (`Approve`) → listo para mezclar.
- **Pedir cambios** (`Request changes`) → arregla y vuelve a pushear.
- **Comentar** (`Comment`) → observaciones sin bloquear.

Cuando pides cambios y pushees commits nuevos a la misma rama, aparecen
automáticamente en el PR. **No tienes que abrir un PR nuevo.** Mismo PR, más
commits.

### 2. Integración continua (CI)

En paralelo con la revisión humana, GitHub ejecuta un robot definido en
`.github/workflows/ci.yml`. Para este repo, el robot corre:

- `make lint` → verifica estilo con `ruff`.
- `make test` → corre los tests rápidos.

Si todo pasa, aparece un **check verde** ✅ al pie del PR. Si falla, un
**check rojo** ❌ con el log del error.

**¿Por qué existe CI?** Porque "en mi máquina funciona" no basta. CI es la
misma receta del `Makefile`, pero corriendo en un servidor neutral de GitHub.
Si tu PR falla en CI pero pasa en tu laptop:

- Olvidaste commitear un archivo nuevo (el servidor no lo tiene).
- Dependes de una variable de entorno o un dato local que no está en el repo.
- Instalaste algo manualmente y no lo añadiste a `pyproject.toml`.

CI es tu segundo par de ojos. Si pasa CI, el proyecto es **reproducible** en
una máquina que no es la tuya. Esa es toda la idea.

---

## Paso 7 — Merge y limpieza

Cuando el PR tiene ✅ aprobación humana y ✅ CI verde, se puede **mergear**
(mezclar a `main`). El botón **Merge pull request** en GitHub lo hace.

Después del merge, en tu máquina:

```bash
git switch main                  # volver a main
git pull                         # traer los cambios recién mergeados
git branch -d feat/mi-primer-cambio   # borrar la rama local (ya se mergeó)
```

Y listo: estás en `main`, actualizada, con un commit más en la historia que
dice `Merge pull request #X from ...`.

Para el próximo cambio, repites desde el [Paso 2](#paso-2--crear-una-rama-para-tu-trabajo).

---

## Comandos que vas a repetir mil veces

```bash
# ¿Dónde estoy?
git status                       # qué hay cambiado, en qué rama
git branch                       # qué ramas tengo, cuál es la actual
git log --oneline -10            # últimos 10 commits (en una línea cada uno)

# Trabajar
git switch main                  # cambiar a main
git switch -c feat/nombre        # crear y cambiar a rama nueva
git switch feat/nombre           # cambiar a rama ya existente

# Commitear
git diff                         # qué líneas cambié y no commiteé
git diff --staged                # qué líneas ya añadí con 'git add'
git add archivo                  # marcar para commit
git commit -m "feat: ..."        # confirmar

# Sincronizar
git pull                         # traer cambios del remoto a mi rama
git push                         # subir mis commits al remoto

# Deshacer
git restore archivo              # deshacer cambios no commiteados en un archivo
git reset HEAD~1                 # deshacer el último commit, dejar los cambios como no-commiteados
```

---

## Problemas comunes

### "Please tell me who you are"

Olvidaste configurar `user.name` y `user.email`. Vuelve al [Paso 0](#paso-0--instalar-y-configurar-git-una-sola-vez).

### "fatal: Authentication failed"

- Si usas HTTPS: tu PAT expiró o no lo generaste. Crea uno nuevo en GitHub.
- Si usas SSH: tu llave SSH no está añadida a GitHub.

### "Your branch is behind 'origin/main'"

Mientras trabajabas, alguien mergeó algo a `main`. Para ponerte al día **en
tu rama**:

```bash
git switch main
git pull
git switch feat/mi-rama
git merge main          # trae los cambios de main a tu rama
```

Si hay conflictos (dos personas tocaron las mismas líneas), git te marca los
archivos. Los abres, decides qué versión queda, haces `git add` del archivo
resuelto y `git commit`.

### "I committed to main by accident"

```bash
git log --oneline -5             # anota el hash del commit anterior al tuyo
git switch -c feat/arreglo        # crea una rama con tu commit
git switch main
git reset --hard <hash-anterior>  # devuelve main a antes de tu commit
```

**No hagas esto si ya pushaste a `main`.** Si ya pushaste, pregunta antes de
intentar nada.

### "Rechazaron mi PR y no sé por qué"

Lee los comentarios en la pestaña **Conversation** del PR, y los checks
rojos al pie. El mensaje de error del CI suele ser muy específico.

---

## Lo que este tutorial no cubre (a propósito)

- **Rebase vs merge.** Para este curso siempre mergeas. Rebase lo verás si
  entras a un equipo que lo prefiera; no es prerrequisito.
- **Submódulos, worktrees, bisect.** Herramientas poderosas que no necesitas
  todavía.
- **Resolución avanzada de conflictos.** Si llegas ahí, pregunta en clase; se
  resuelve mejor con alguien al lado.

---

## Siguiente paso

Con git listo, el flujo del curso es:

- **Tutorial 02 (`02-configuracion-proyecto-uv.md`)** → cómo se estructura un proyecto Python con `uv`.
- **Tutorial 02c (`02c-el-makefile-del-curso.md`)** → por qué automatizamos comandos con `make`.

Vuelve al [índice de tutoriales](README.md) para ver la ruta completa.
