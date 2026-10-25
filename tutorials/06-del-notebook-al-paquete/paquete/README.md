# Paquete `analisis-ventas` (estado 3 del tutorial)

Paquete minimo que acompaña al tutorial. Se instala en modo editable:

```bash
uv pip install -e .
```

Y luego se usa como cualquier otro paquete Python:

```python
from analisis_ventas import limpiar, precio_promedio_ponderado
from analisis_ventas.config import PRECIO_MIN, PRECIO_MAX
```

Tests:

```bash
cd ..  # vuelve al tutorial
pytest paquete/tests/
```
