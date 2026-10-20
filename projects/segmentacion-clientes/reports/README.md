# reports/ — Artefactos de referencia del caso de recompra

Esta carpeta guarda los **outputs** del caso guía: cifras y cohortes que las
guías del material citan textualmente. No son fixtures de pytest (los tests
`-m slow` regeneran las cohortes desde el parquet crudo); son el "qué número
tiene que salir" al lado del código, versionado para que el estudiante abra
el repo y lo vea sin descargar nada.

## Qué hay

| Archivo | Qué contiene | Tamaño |
|---|---|---|
| `resultados.json` | Métricas de los 4 modelos sobre las cohortes A y B (AUC, AP, F1, precision, recall en CV y test), más tamaño y tasa positiva de cada cohorte | ~2 KB |
| `cohorte_A.parquet` | Cohorte de corte A con la variable `recompra` y las features RFM | ~63 KB |
| `cohorte_B.parquet` | Cohorte de corte B con la variable `recompra` y las features RFM | ~108 KB |
| `figures/` | Figuras usadas en los `.md` de sesión. Los PNG/HTML están ignorados; solo `.gitkeep` se versiona |
| `perfil_muestra.html` | Reporte auto-generado por `fg-data-profiling`. **No se versiona** (ignorado en `.gitignore`) |

## Cómo se regeneran

```bash
# 1. Descarga el parquet crudo (Online Retail II, UCI 502). Hace falta una sola vez.
uv run python projects/segmentacion-clientes/scripts/descargar_datos.py

# 2. Corre el caso de recompra. Reescribe los tres archivos de arriba.
uv run python projects/segmentacion-clientes/scripts/caso_recompra.py
```

Si una cifra de `resultados.json` cambia, hay dos opciones honestas:

1. **Alguna lógica cambió** (preprocesador, split, modelo): se actualiza la
   `docs/specs/` correspondiente y los `.md` de sesión que la citen.
2. **Nada debía cambiar**: hay un bug. Se investiga antes de aceptar la cifra.

Nunca se maquilla el JSON para que coincida con el código.
