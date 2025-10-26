# Diccionario de Datos - Dataset MercadoLibre

## Descripción de Variables

A continuación se presenta una breve descripción de las variables disponibles en el dataset `MLA_100k.jsonlines`.

### Variables Principales

| Variable | Descripción en Español |
|----------|------------------------|
| `seller_address` | Dirección del vendedor |
| `warranty` | Información sobre la garantía del producto |
| `sub_status` | NO CLARO - Parece estar relacionado con cuando una publicación de producto está suspendida (?) |
| `condition` | **[VARIABLE OBJETIVO]** Condición del producto (`new` para nuevo, `used` para usado) |
| `seller_contact` | Información de contacto del vendedor |
| `deal_ids` | Lista de IDs de ofertas relacionadas con el producto |
| `base_price` | Precio base del producto antes de descuentos |
| `shipping` | Información sobre el método de envío |
| `non_mercado_pago_payment_methods` | Métodos de pago que no son Mercado Pago |
| `seller_id` | ID único del vendedor |
| `variations` | Contiene información sobre variaciones del producto (color, talla, etc.) |
| `location` | Información sobre la ubicación del producto o del vendedor |
| `site_id` | Código del sitio donde se está vendiendo el producto (MLA para Argentina, MLB para Brasil, etc.) |
| `listing_type_id` | Tipo de publicación (gold_special, gold_pro, free, etc.) |
| `price` | Precio actual del producto (¿Cuál es la diferencia con base_price?) |
| `attributes` | Atributos del producto (marca, peso, dimensiones, modelo, etc.) |
| `buying_mode` | Modo de compra |
| `tags` | Lista de etiquetas asociadas con el producto |
| `parent_item_id` | ID del producto principal si pertenece a una serie de variantes |
| `coverage_areas` | NO CLARO - Posiblemente áreas de cobertura de envío |
| `category_id` | ID de la categoría del producto |
| `descriptions` | Descripción del producto con alguna característica especial (¿cuál?) |
| `international_delivery_mode` | Envío internacional |
| `pictures` | Lista de imágenes asociadas con el producto |
| `id` | ID único del producto |
| `official_store_id` | ID de la tienda oficial |
| `accepts_mercadopago` | El vendedor acepta Mercado Pago |
| `original_price` | Precio original antes de aplicar descuentos (¿CUÁL ES LA DIFERENCIA con base_price?) |
| `currency_id` | Moneda en la que está listado el precio |
| `thumbnail` | URL de la imagen miniatura del producto |
| `title` | Nombre o título del producto |
| `automatic_relist` | NO CLARO - Posiblemente republicación automática del producto |
| `date_created` | Fecha en que se creó la publicación |
| `stop_time` | Fecha en que se desactiva la publicación |
| `status` | Estado de la publicación |
| `catalog_product_id` | ID del producto en el catálogo de MercadoLibre |
| `initial_quantity` | Cantidad inicial de productos disponibles |
| `sold_quantity` | Número de unidades vendidas |
| `available_quantity` | Número de unidades aún disponibles |

---

## Notas Importantes

### Variables con Ambigüedad

Algunas variables tienen significados que requieren clarificación adicional:

1. **Diferencias de Precio**:
   - `price`: Precio actual mostrado al comprador
   - `base_price`: Precio base antes de descuentos
   - `original_price`: Precio original antes de descuentos

   **¿Cuál es la diferencia?** Requiere investigación en el EDA para determinar cuándo difieren.

2. **Variables No Claras**:
   - `sub_status`: Posiblemente relacionado con suspensiones de publicaciones
   - `coverage_areas`: Posiblemente áreas geográficas de cobertura de envío
   - `automatic_relist`: Posiblemente indica si la publicación se renueva automáticamente
   - `descriptions`: Estructura exacta y características especiales por determinar

### Variable Objetivo

**`condition`**: Esta es la variable que se desea predecir en el modelo de clasificación.

- Valores posibles: `"new"` (nuevo) o `"used"` (usado)
- Tipo: Categórica binaria
- Importancia: Variable objetivo del modelo de Machine Learning

---

## Campos Anidados Importantes

### `seller_address` (Objeto)

```json
{
  "country": {"id": "AR", "name": "Argentina"},
  "state": {"id": "AR-B", "name": "Buenos Aires"},
  "city": {"id": "TUxBQkNBUzQzMjM", "name": "Capital Federal"}
}
```

**Features derivadas comunes**:
- `seller_country`
- `seller_state`
- `seller_city`

### `shipping` (Objeto)

```json
{
  "mode": "me2",
  "local_pick_up": true,
  "free_shipping": false,
  "tags": ["fulfillment", "mandatory_free_shipping"]
}
```

**Features derivadas comunes**:
- `shipping_mode`
- `shipping_local_pick_up`
- `shipping_free_shipping`
- `shipping_tags` (one-hot encoding)

### `attributes` (Array de Objetos)

```json
[
  {"id": "BRAND", "name": "Marca", "value_name": "Samsung"},
  {"id": "MODEL", "name": "Modelo", "value_name": "Galaxy S21"}
]
```

**Features derivadas comunes**:
- Extracción de marca (`BRAND`)
- Extracción de modelo (`MODEL`)
- Otros atributos según categoría de producto

---

## Uso en Feature Engineering

### Tipos de Variables para ML

| Tipo | Variables | Estrategia de Encoding |
|------|-----------|----------------------|
| **Numéricas** | `price`, `initial_quantity`, `sold_quantity`, `available_quantity` | Normalización/Estandarización |
| **Categóricas Ordinales** | `listing_type_id`, `condition` | Label Encoding |
| **Categóricas Nominales** | `site_id`, `currency_id`, `category_id` | One-Hot Encoding |
| **Booleanas** | `accepts_mercadopago`, `shipping.free_shipping` | Convertir a 0/1 |
| **Temporales** | `date_created`, `stop_time` | Extracción de componentes (año, mes, día, día de la semana) |
| **Texto** | `title`, `warranty` | TF-IDF, Regex, Conteo de palabras |
| **Anidadas** | `seller_address`, `shipping`, `attributes` | Aplanar y extraer campos relevantes |

---

## Referencias

- [Documentación API MercadoLibre](https://developers.mercadolibre.com/)
- [Estructura de respuesta de productos](https://developers.mercadolibre.com/es_ar/items-y-busquedas)

---

**Última actualización**: 2025-10-25
**Autor**: David Palacio Jiménez
