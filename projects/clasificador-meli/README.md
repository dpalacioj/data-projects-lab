# 🛒 Clasificador de Productos Mercado Libre

Proyecto de clasificación de productos de Mercado Libre utilizando técnicas de Machine Learning.

---

## 📦 Dataset

El dataset para este proyecto se almacena usando **Git LFS (Large File Storage)** debido a su tamaño (~316 MB).

### ¿Qué es Git LFS?

Git LFS es una extensión de Git que permite versionar archivos grandes sin saturar el repositorio. Los archivos grandes se almacenan en un servidor externo y Git solo guarda referencias pequeñas.

---

## 🚀 Configuración de Git LFS (Primera Vez)

### Paso 1: Instalar Git LFS

#### macOS:
```bash
brew install git-lfs
```

#### Linux (Ubuntu/Debian):
```bash
sudo apt-get install git-lfs
```

#### Windows:
Descarga el instalador desde: https://git-lfs.github.com/

---

### Paso 2: Inicializar Git LFS en tu Usuario

```bash
# Solo necesitas hacer esto UNA VEZ por usuario
git lfs install
```

Deberías ver:
```
✅ Updated git hooks.
✅ Git LFS initialized.
```

---

### Paso 3: Clonar el Repositorio con LFS

Si **aún no has clonado el repositorio:**

```bash
# Clonar normalmente (LFS se activa automáticamente)
git clone https://github.com/dpalacioj/data-projects-lab.git
cd data-projects-lab
```

Si **ya tienes el repositorio clonado:**

```bash
cd data-projects-lab

# Descargar archivos LFS
git lfs pull
```

---

## 📥 Descarga del Dataset

Después de configurar Git LFS, el dataset se descarga automáticamente:

```bash
# Verificar que el archivo existe
ls -lh datos/*.jsonl

# Debería mostrar algo como:
# -rw-r--r-- 1 user group 316M Oct 20 2025 datos/meli_clasificacion.jsonl
```

Si el archivo **NO** está o es muy pequeño (<1KB):

```bash
# Forzar descarga de archivos LFS
git lfs pull
```

---

## 🔍 Verificar Configuración

```bash
# Ver qué archivos están en LFS
git lfs ls-files

# Debería mostrar:
# 1mXW0DwSH... - datos/meli_clasificacion.jsonl
```

---

## 📊 Uso del Dataset en Notebooks

```python
import pandas as pd

# Leer el dataset
df = pd.read_json('../../datos/meli_clasificacion.jsonl', lines=True)

print(f"Dataset: {df.shape[0]:,} filas × {df.shape[1]} columnas")
```

---

## ⚠️ Notas Importantes

1. **Límites de GitHub LFS (Cuenta Gratuita):**
   - 1 GB de almacenamiento
   - 1 GB de ancho de banda por mes
   - Si excedes, el repo sigue funcionando pero las descargas LFS se pausan

2. **No commitear archivos grandes sin LFS:**
   ```bash
   # ❌ MAL - Archivo grande sin LFS
   git add datos/archivo_grande.csv

   # ✅ BIEN - Primero trackear con LFS
   git lfs track "datos/*.csv"
   git add .gitattributes
   git add datos/archivo_grande.csv
   ```

3. **Archivos ya trackeados con LFS:**
   - `*.jsonl` (JSON Lines)
   - `*.parquet`
   - `*.csv` (>10MB)

---

## 🛠️ Para Contribuidores

### Si necesitas agregar archivos grandes:

```bash
# 1. Trackear el tipo de archivo con LFS
git lfs track "datos/*.nuevo_formato"

# 2. Agregar el .gitattributes actualizado
git add .gitattributes

# 3. Agregar tu archivo
git add datos/mi_archivo.nuevo_formato

# 4. Commit normal
git commit -m "feat: agregar nuevo dataset"

# 5. Push (LFS se encarga automáticamente)
git push
```

---

## 📚 Estructura del Proyecto

```
projects/clasificador-meli/
├── README.md                    # Este archivo
├── 01_descarga_datos.ipynb      # (Próximamente)
├── 02_eda.ipynb                 # (Próximamente)
└── ...
```

**Dataset:** `datos/meli_clasificacion.jsonl` (en raíz del repo)

---

## 🆘 Solución de Problemas

### Problema: "El archivo .jsonl es muy pequeño (1KB)"

**Causa:** Git descargó solo el puntero LFS, no el archivo real.

**Solución:**
```bash
git lfs pull
```

---

### Problema: "git lfs: command not found"

**Causa:** Git LFS no está instalado.

**Solución:**
```bash
# macOS
brew install git-lfs

# Linux
sudo apt-get install git-lfs

# Luego inicializar
git lfs install
```

---

### Problema: "Bandwidth limit exceeded"

**Causa:** Excediste el límite mensual de 1GB de GitHub LFS.

**Solución:**
- Espera al siguiente mes
- O descarga el dataset manualmente desde: [Google Drive](https://drive.google.com/file/d/1mXW-0DwSHX0sSklp3lQxLChw3XDDwM1b/view) (backup)

---

## 📝 Recursos

- [Documentación oficial Git LFS](https://git-lfs.github.com/)
- [GitHub: About Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)
- [Git LFS Tutorial](https://www.atlassian.com/git/tutorials/git-lfs)

---

## 👤 Autor

**David Palacio Jiménez**

- 📧 Email: davidpalacioj@gmail.com
- 🐙 GitHub: [dpalacioj](https://github.com/dpalacioj)

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver [LICENSE](../../LICENSE) para más detalles.

**Copyright (c) 2025 David Palacio Jiménez**
