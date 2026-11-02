---
title: "01 · Git y GitHub"
parent: Tutoriales
nav_order: 1
---
# Tutorial — Git y GitHub: el flujo de trabajo del curso

Última revisión: 2026-10-31. Prerrequisito: una computadora con conexión a
internet. Todo lo demás (cuenta de GitHub, git instalado, autenticación) lo
haces dentro de este tutorial.

Este tutorial **no** es una referencia completa de git. Es la ruta mínima para
que puedas:

1. Clonar este repositorio.
2. Trabajar en una rama propia sin romper el trabajo de nadie.
3. Hacer tu primer **Pull Request** (PR) y entender qué pasa cuando lo abres.

Si nunca has usado git, empieza aquí. Si ya has hecho `git clone` alguna vez
pero nunca un PR, también. Si ya trabajas con PRs todos los días, puedes saltar
al [cheatsheet de comandos](#comandos-que-vas-a-repetir-mil-veces).

Las instrucciones de instalación y autenticación están sacadas de la
documentación oficial:

- Instalación de git: [Pro Git · Installing Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
- Configuración inicial: [Pro Git · First-Time Git Setup](https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup).
- Cuenta en GitHub: [GitHub Docs · Creating an account on GitHub](https://docs.github.com/en/get-started/start-your-journey/creating-an-account-on-github).
- Autenticación: [GitHub Docs · Caching your GitHub credentials in Git](https://docs.github.com/en/get-started/git-basics/caching-your-github-credentials-in-git)
  y [Managing your personal access tokens](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

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

## Antes de empezar: cuenta de GitHub y terminal

Antes del Paso 0 hay dos cosas que no son git pero que git necesita.

### Crear tu cuenta en GitHub

Si ya tienes cuenta, salta a la siguiente sección.

Fuente: [GitHub Docs · Creating an account on GitHub](https://docs.github.com/en/get-started/start-your-journey/creating-an-account-on-github).

1. Entra a <https://github.com/signup>.
2. Sigue las indicaciones del formulario (correo, contraseña, nombre de
   usuario). El **nombre de usuario** aparecerá en tus commits y en la URL de
   tu perfil (`github.com/<tu-usuario>`); elígelo pensando que es público y
   que no vale la pena cambiarlo después.
3. GitHub te envía un código al correo. Introdúcelo para **verificar el
   email**. Si no verificas, no puedes hacer push.
4. GitHub recomienda **activar 2FA (autenticación en dos factores)** desde el
   primer día. Hazlo cuando puedas, siguiendo la [guía oficial de
   2FA](https://docs.github.com/en/authentication/securing-your-account-with-two-factor-authentication-2fa/configuring-two-factor-authentication).

Usa un correo que vayas a conservar. Más adelante lo usas en `git config` y
tiene que coincidir con el que registraste en GitHub para que los commits
queden ligados a tu perfil.

### Qué es una terminal y cómo abrirla

Todas las instrucciones de git se escriben en una **terminal** (también
llamada "consola" o "línea de comandos"). Es una ventana donde escribes
texto, presionas <kbd>Enter</kbd> y el sistema operativo responde con más
texto. Nada de botones.

| SO | Qué abrir | Cómo |
|---|---|---|
| macOS | **Terminal** | <kbd>⌘</kbd> + <kbd>Espacio</kbd> → escribe `Terminal` → <kbd>Enter</kbd> |
| Linux (Ubuntu/Debian) | **Terminal** o **GNOME Terminal** | <kbd>Ctrl</kbd> + <kbd>Alt</kbd> + <kbd>T</kbd> |
| Windows | **Git Bash** (lo instala Git for Windows en el Paso 0) | Menú Inicio → `Git Bash` |

En Windows **recomendamos Git Bash** y no PowerShell ni CMD. Git Bash
reproduce el entorno Unix que usan macOS y Linux, así que todos los comandos
de este tutorial funcionan igual en los tres sistemas.

Cuando una instrucción diga "en la terminal, ejecuta…" significa: copia la
línea, pégala en esta ventana y presiona <kbd>Enter</kbd>. El símbolo `$` o
`#` al inicio en algunos ejemplos **no se copia**: es el *prompt* del
sistema, no parte del comando.

Si nunca has navegado con la terminal, lee primero el
[tutorial 00 — Ambientes](00-ambientes-venv-conda-uv/) donde se cubre `cd`,
`ls`/`dir`, `pwd` y el `~` (tu carpeta de usuario).

---

## Paso 0 — Instalar y configurar git (una sola vez)

### 0.1 Instalar git

Fuente: [Pro Git · Installing Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

Primero comprueba si ya lo tienes. Abre la terminal y ejecuta:

```bash
git --version
```

- Si responde algo como `git version 2.46.0`, ya está instalado: salta a
  [0.2](#02-verificar-la-instalación).
- Si responde "command not found" o similar, sigue las instrucciones de tu
  sistema operativo abajo.

#### macOS

En macOS 10.9 (Mavericks) o posterior, al ejecutar `git --version` por
primera vez, el sistema abre un cuadro de diálogo que ofrece instalar las
**Xcode Command Line Tools** (incluyen git). Acepta.

Si no aparece el diálogo, fuerza la instalación con:

```bash
xcode-select --install
```

Alternativa oficial: descargar el instalador desde
<https://git-scm.com/download/mac>.

#### Linux — Debian / Ubuntu

```bash
sudo apt update
sudo apt install git-all
```

`sudo` te pedirá tu contraseña de usuario.

#### Linux — Fedora / RHEL / CentOS

```bash
sudo dnf install git-all
```

Para otras distribuciones: <https://git-scm.com/download/linux>.

#### Windows

Descarga el instalador oficial **Git for Windows** desde
<https://git-scm.com/download/win> y ejecútalo.

El asistente hace muchas preguntas. Para este curso, en cada pantalla
**deja los valores por defecto** salvo donde se indique lo contrario:

| Pantalla | Qué hacer |
|---|---|
| *Select Components* | Deja marcadas las opciones por defecto (incluye **Git Bash Here** y **Git GUI Here**). |
| *Choosing the default editor* | Elige **"Use Visual Studio Code as Git's default editor"** si tienes VS Code; si no, deja Nano. **Evita vim** si no sabes salir de vim. |
| *Adjusting the name of the initial branch* | Elige **"Override the default branch name for new repositories"** y escribe `main`. |
| *Adjusting your PATH environment* | Deja la opción recomendada: **"Git from the command line and also from 3rd-party software"**. |
| *Choosing HTTPS transport backend* | Deja **OpenSSL**. |
| *Configuring the line ending conversions* | Deja **"Checkout Windows-style, commit Unix-style line endings"**. |
| *Configuring the terminal emulator* | Deja **MinTTY**. |
| *Credential helper* | Deja **Git Credential Manager** (es lo que te evita escribir el token en cada push). |

El instalador deja disponible **Git Bash** en el menú Inicio. Úsalo para
todo lo que diga "terminal" en este tutorial.

### 0.2 Verificar la instalación

Cierra y vuelve a abrir la terminal (así toma los cambios del PATH) y
ejecuta:

```bash
git --version
```

Debe imprimir una versión (por ejemplo `git version 2.46.0`). Si todavía
dice "command not found":

- **macOS/Linux**: cierra y abre una terminal nueva.
- **Windows**: usa **Git Bash**, no PowerShell ni CMD.

### 0.3 Configurar tu identidad

Fuente: [Pro Git · First-Time Git Setup](https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup).

Git firma cada commit con tu nombre y correo. Esto se hace **una sola vez**
por máquina:

```bash
git config --global user.name "Tu Nombre"
git config --global user.email "tu-correo@dominio.com"
```

Reglas:

- Usa el **mismo correo** que registraste en GitHub. Si no coincide, los
  commits se guardan pero no aparecen asociados a tu perfil en la web de
  GitHub.
- El nombre puede llevar espacios (por eso van las comillas).
- `--global` guarda la configuración en tu carpeta de usuario (`~/.gitconfig`),
  así aplica a todos los repositorios de esta máquina.

### 0.4 Configurar tu editor por defecto (así no te atrapa vim)

Algunos comandos de git (`git commit` sin `-m`, `git rebase -i`, resolver
mergeos) abren un editor para que escribas texto. Si no configuras uno, en
Linux y macOS abre **vim** por defecto, un editor que es muy útil pero que
tiene fama de ser difícil de cerrar la primera vez.

Elige según lo que ya tengas instalado:

```bash
# VS Code (recomendado si ya lo usas)
git config --global core.editor "code --wait"

# Nano (viene preinstalado en casi todo Linux y macOS, fácil de usar)
git config --global core.editor "nano"
```

Si no cambias nada y acabas dentro de vim por accidente: presiona
<kbd>Esc</kbd>, escribe `:wq` y pulsa <kbd>Enter</kbd> para guardar y salir;
o `:q!` para salir sin guardar.

### 0.5 Configurar el nombre de rama por defecto

Para que al crear repositorios locales la rama inicial se llame `main` (y no
`master`, el nombre antiguo):

```bash
git config --global init.defaultBranch main
```

Esto solo afecta a repos que crees con `git init`. Al clonar este
repositorio ya traes la rama con el nombre correcto.

### 0.6 Revisar tu configuración

Comprueba que quedó todo bien:

```bash
git config --list
```

Deberías ver tu `user.name`, `user.email`, `core.editor` e
`init.defaultBranch`. Para ver un valor concreto:

```bash
git config user.name
git config user.email
```

### 0.7 Autenticación con GitHub

Al hacer `git push` por primera vez GitHub te va a pedir que demuestres que
eres tú. **No sirve la contraseña de la web** desde 2021. Hay dos caminos
oficiales:

- **Opción A — HTTPS + Git Credential Manager + Personal Access Token (PAT).**
  Lo recomendado para empezar en este curso. Guardas un token una vez y el
  sistema operativo lo almacena de forma segura para los siguientes push.
- **Opción B — SSH.** Más cómodo a largo plazo (no vence, no tienes que
  renovar tokens). Requiere generar una llave y pegarla en GitHub.

Elige **una**. Puedes cambiar de método después.

#### Opción A — HTTPS + Git Credential Manager + PAT

Fuente: [GitHub Docs · Caching your GitHub credentials in Git](https://docs.github.com/en/get-started/git-basics/caching-your-github-credentials-in-git).

**A.1 Instalar Git Credential Manager (GCM)**

- **Windows**: ya viene con Git for Windows si dejaste la opción por defecto
  en el instalador. No haces nada.
- **macOS**:
  ```bash
  brew install --cask git-credential-manager
  ```
  (Requiere [Homebrew](https://brew.sh/) instalado.)
- **Linux**: sigue la [guía oficial de instalación de GCM](https://github.com/git-ecosystem/git-credential-manager/blob/main/docs/install.md).
  Hay paquete `.deb` para Debian/Ubuntu y `.rpm` para Fedora.

**A.2 Generar un Personal Access Token (PAT)**

Fuente: [GitHub Docs · Managing your personal access tokens](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

Para el curso basta con un **classic PAT**:

1. En GitHub, click en tu foto (arriba a la derecha) → **Settings**.
2. En la barra lateral izquierda, baja hasta **Developer settings**.
3. **Personal access tokens** → **Tokens (classic)**.
4. **Generate new token** → **Generate new token (classic)**.
5. Rellena:
   - **Note**: una descripción, por ejemplo `laptop-curso-2026`.
   - **Expiration**: 90 días es un buen compromiso. Al vencer, generas otro.
   - **Scopes**: marca **`repo`** (es el permiso que habilita `git push` a
     repos que te pertenecen o donde colaboras).
6. **Generate token**. GitHub muestra el token **una sola vez**. Cópialo.

Guárdalo temporalmente en algún lugar seguro (gestor de contraseñas). Si lo
pierdes, no se puede recuperar: generas otro.

**A.3 Usar el token**

La primera vez que hagas `git push` (más adelante en este tutorial), GCM
abre una ventana del navegador para autenticarte, o te pide usuario y
"contraseña" en la terminal. En "contraseña" pegas el PAT que acabas de
generar. GCM lo guarda en el almacén seguro del sistema (Keychain en macOS,
Credential Manager en Windows, libsecret en Linux) y no te lo vuelve a
pedir mientras no venza.

#### Opción B — SSH

Fuente: [GitHub Docs · Generating a new SSH key and adding it to the ssh-agent](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

**B.1 Generar la llave (los tres sistemas)**

```bash
ssh-keygen -t ed25519 -C "tu-correo@dominio.com"
```

Cuando pregunte dónde guardarla, presiona <kbd>Enter</kbd> para aceptar la
ruta por defecto (`~/.ssh/id_ed25519`). Cuando pida *passphrase*, pon una
contraseña y guárdala; o deja vacío si prefieres sin contraseña (menos
seguro pero más cómodo).

**B.2 Añadir la llave al ssh-agent**

*macOS*:

```bash
eval "$(ssh-agent -s)"
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
```

*Linux*:

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

*Windows (PowerShell como administrador la primera vez para habilitar el servicio)*:

```powershell
Get-Service -Name ssh-agent | Set-Service -StartupType Manual
Start-Service ssh-agent
ssh-add $HOME\.ssh\id_ed25519
```

**B.3 Añadir la llave pública a GitHub**

Copia el contenido de la llave **pública** (termina en `.pub`):

```bash
# macOS
pbcopy < ~/.ssh/id_ed25519.pub

# Linux (requiere xclip)
xclip -selection clipboard < ~/.ssh/id_ed25519.pub

# Windows (Git Bash)
cat ~/.ssh/id_ed25519.pub | clip
```

En GitHub: foto → **Settings** → **SSH and GPG keys** → **New SSH key**.
Pega el contenido, ponle un título descriptivo (p. ej. `laptop-personal`),
y guarda.

Verifica la conexión:

```bash
ssh -T git@github.com
```

Debe responder: *"Hi `<tu-usuario>`! You've successfully authenticated…"*.

Si usas SSH, al clonar en el [Paso 1](#paso-1--clonar-el-repositorio) usa la
URL `git@github.com:...` en vez de la de HTTPS.

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

- **Tutorial 01b** ([`01b-tu-primer-ci.md`](01b-tu-primer-ci.md)) → construye tu propio repo con CI desde cero. Es el que termina de aterrizar qué pasa cuando abres un PR.
- **Tutorial 01c** ([`01c-tu-pagina-con-github-pages.md`](01c-tu-pagina-con-github-pages.md)) → publica tu repo como sitio web con GitHub Pages (opcional pero útil para el entregable).
- **Tutorial 02** ([`02-configuracion-proyecto-uv.md`](02-configuracion-proyecto-uv.md)) → cómo se estructura un proyecto Python con `uv`.
- **Tutorial 03** ([`03-el-makefile-del-curso.md`](03-el-makefile-del-curso.md)) → por qué automatizamos comandos con `make`.

Vuelve al [índice de tutoriales](README.md) para ver la ruta completa.
