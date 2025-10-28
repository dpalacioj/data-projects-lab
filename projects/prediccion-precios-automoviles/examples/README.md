# Ejemplos de Uso de la API

Esta carpeta contiene ejemplos para probar la API de predicción de precios de automóviles.

## 📋 Prerequisitos

**Inicia la API primero:**

```bash
cd prediccion-precios-automoviles
python app_fastapi.py
```

La API debería estar corriendo en: http://localhost:8000

## 🛠️ Opciones para Probar la API

### **Opción 1: Interfaz Interactiva de FastAPI** (Más Fácil)

1. Abre en tu navegador: http://localhost:8000/docs
2. Verás la documentación interactiva (Swagger UI)
3. Click en cualquier endpoint
4. Click en "Try it out"
5. Llena los parámetros
6. Click en "Execute"

**Ventajas:**
- ✅ No necesitas instalar nada
- ✅ Muy visual e intuitivo
- ✅ Muestra el schema de los datos
- ✅ Ideal para principiantes

---

### **Opción 2: Script de Python** (Recomendado)

**Archivo:** `test_api.py`

```bash
# Instalar requests si no lo tienes
pip install requests

# Ejecutar script
python examples/test_api.py
```

**Ventajas:**
- ✅ Ejemplos predefinidos listos para usar
- ✅ Muestra resultados formateados
- ✅ Fácil de modificar para tus propios casos
- ✅ Aprende a usar Python requests

**El script prueba:**
- Health check
- Información del modelo
- 5 predicciones con diferentes tipos de autos

---

### **Opción 3: Archivo .http** (Para VSCode)

**Archivo:** `api_requests.http`

**Pasos:**

1. Instala la extensión **"REST Client"** en VSCode
2. Abre el archivo `examples/api_requests.http`
3. Verás botones "Send Request" sobre cada petición
4. Click en "Send Request" para ejecutar

**Ventajas:**
- ✅ Múltiples requests en un solo archivo
- ✅ Fácil de organizar y versionar
- ✅ Sintaxis clara
- ✅ Ideal para desarrollo

---

### **Opción 4: Script Bash con cURL** (Para Terminal)

**Archivo:** `curl_examples.sh`

```bash
# Dar permisos de ejecución
chmod +x examples/curl_examples.sh

# Ejecutar todo el script
bash examples/curl_examples.sh
```

**O copiar comandos individuales:**

```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Predicción
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "marca": "Toyota",
    "tipo_carroceria": "SUV",
    "año": 2020,
    "kilometraje": 50000,
    "tipo_combustible": "Gasolina",
    "transmision": "Automática",
    "cilindrada": 2000,
    "potencia": 150,
    "peso": 1500,
    "consumo": 8.5,
    "color": "Blanco",
    "edad_propietarios": 1,
    "calificacion_estado": 8.5,
    "region_venta": "Centro"
  }'
```

**Ventajas:**
- ✅ Funciona en cualquier terminal
- ✅ No requiere dependencias adicionales
- ✅ Útil para scripts y automation
- ✅ Estándar de la industria

---

## 📊 Casos de Prueba Incluidos

### 1. **Toyota SUV 2020** - Auto estándar familiar
- Precio esperado: ~$18,000 - $20,000

### 2. **BMW Sedán 2022** - Auto de lujo
- Precio esperado: ~$35,000 - $45,000

### 3. **Honda Hatchback 2018** - Auto económico
- Precio esperado: ~$10,000 - $12,000

### 4. **Mercedes-Benz SUV 2023** - Auto premium nuevo
- Precio esperado: ~$50,000 - $65,000

### 5. **Ford Pickup 2015** - Auto de trabajo
- Precio esperado: ~$12,000 - $15,000

---

## 🎯 Endpoints Disponibles

### `GET /health`
Verifica que la API está funcionando.

**Respuesta:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "Servicio operativo..."
}
```

---

### `GET /model-info`
Obtiene información sobre el modelo cargado.

**Respuesta:**
```json
{
  "model_type": "LGBMRegressor",
  "features": [...],
  "metrics": {...}
}
```

---

### `POST /predict`
Realiza una predicción de precio.

**Body (JSON):**
```json
{
  "marca": "Toyota",
  "tipo_carroceria": "SUV",
  "año": 2020,
  "kilometraje": 50000,
  "tipo_combustible": "Gasolina",
  "transmision": "Automática",
  "cilindrada": 2000,
  "potencia": 150,
  "peso": 1500,
  "consumo": 8.5,
  "color": "Blanco",
  "edad_propietarios": 1,
  "calificacion_estado": 8.5,
  "region_venta": "Centro"
}
```

**Respuesta:**
```json
{
  "precio_predicho": 18500.00,
  "precio_formateado": "$18,500.00 USD",
  "modelo_usado": "LGBMRegressor",
  "status": "success"
}
```

---

## 🧪 Crear Tus Propios Casos de Prueba

### Valores Válidos:

**Marcas:**
- Toyota, Honda, Ford, Chevrolet, Nissan, Mazda, Hyundai, Kia, BMW, Mercedes-Benz, Audi

**Tipo de Carrocería:**
- Sedán, SUV, Hatchback, Pickup, Coupé, Minivan

**Tipo de Combustible:**
- Gasolina, Diesel, Híbrido, Eléctrico

**Transmisión:**
- Manual, Automática

**Colores:**
- Blanco, Negro, Gris, Plata, Rojo, Azul, Verde, Amarillo

**Regiones:**
- Norte, Sur, Este, Oeste, Centro

**Rangos Numéricos:**
- Año: 2010-2024
- Kilometraje: 0-500,000
- Cilindrada: 1000-5000 cc
- Potencia: 50-500 HP
- Peso: 800-3000 kg
- Consumo: 0-20 L/100km (0 para eléctricos)
- Edad Propietarios: 1-10
- Calificación Estado: 1-10

---

## ⚠️ Solución de Problemas

### Error: "Connection refused"
**Causa:** La API no está corriendo
**Solución:** Inicia la API con `python app_fastapi.py`

### Error: "Model not loaded"
**Causa:** El modelo no fue entrenado
**Solución:** Ejecuta primero `notebooks/02_training.ipynb`

### Error: 422 Unprocessable Entity
**Causa:** Datos inválidos en el request
**Solución:** Verifica que todos los campos tengan valores válidos

---

## 📚 Recursos Adicionales

- **Documentación Interactiva:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/health

---

## 💡 Tips

1. **Usa Swagger UI primero** si eres principiante
2. **Usa el script Python** para automatizar pruebas
3. **Usa .http files** para desarrollo diario en VSCode
4. **Usa cURL** para scripts de CI/CD o automation

¡Diviértete probando la API! 🚀
