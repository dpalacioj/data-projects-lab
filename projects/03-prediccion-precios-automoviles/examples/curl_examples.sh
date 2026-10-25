#!/bin/bash
# Ejemplos de cURL para probar la API de Predicción de Precios
#
# Uso:
#   1. Inicia la API: python app_fastapi.py
#   2. Ejecuta estos comandos uno por uno, o todo el script: bash curl_examples.sh

API_URL="http://localhost:8000"

echo "=========================================="
echo "  Probando API de Predicción de Precios"
echo "=========================================="
echo ""

# 1. Health Check
echo "1️⃣  Health Check"
echo "----------------------------------------"
curl -s -X GET "$API_URL/health" | python3 -m json.tool
echo ""
echo ""

# 2. Información del modelo
echo "2️⃣  Información del Modelo"
echo "----------------------------------------"
curl -s -X GET "$API_URL/model-info" | python3 -m json.tool
echo ""
echo ""

# 3. Predicción - Toyota SUV 2020
echo "3️⃣  Predicción: Toyota SUV 2020"
echo "----------------------------------------"
curl -s -X POST "$API_URL/predict" \
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
  }' | python3 -m json.tool
echo ""
echo ""

# 4. Predicción - BMW Sedán 2022
echo "4️⃣  Predicción: BMW Sedán 2022 (Lujo)"
echo "----------------------------------------"
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "marca": "BMW",
    "tipo_carroceria": "Sedán",
    "año": 2022,
    "kilometraje": 20000,
    "tipo_combustible": "Híbrido",
    "transmision": "Automática",
    "cilindrada": 2500,
    "potencia": 250,
    "peso": 1700,
    "consumo": 6.5,
    "color": "Negro",
    "edad_propietarios": 1,
    "calificacion_estado": 9.5,
    "region_venta": "Norte"
  }' | python3 -m json.tool
echo ""
echo ""

# 5. Predicción - Honda Hatchback 2018
echo "5️⃣  Predicción: Honda Hatchback 2018 (Económico)"
echo "----------------------------------------"
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "marca": "Honda",
    "tipo_carroceria": "Hatchback",
    "año": 2018,
    "kilometraje": 80000,
    "tipo_combustible": "Gasolina",
    "transmision": "Manual",
    "cilindrada": 1500,
    "potencia": 100,
    "peso": 1200,
    "consumo": 7.0,
    "color": "Azul",
    "edad_propietarios": 2,
    "calificacion_estado": 7.0,
    "region_venta": "Sur"
  }' | python3 -m json.tool
echo ""
echo ""

# 6. Predicción - Ford Pickup 2015
echo "6️⃣  Predicción: Ford Pickup 2015 (Trabajo)"
echo "----------------------------------------"
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "marca": "Ford",
    "tipo_carroceria": "Pickup",
    "año": 2015,
    "kilometraje": 150000,
    "tipo_combustible": "Diesel",
    "transmision": "Manual",
    "cilindrada": 3000,
    "potencia": 180,
    "peso": 2200,
    "consumo": 10.0,
    "color": "Blanco",
    "edad_propietarios": 3,
    "calificacion_estado": 6.0,
    "region_venta": "Oeste"
  }' | python3 -m json.tool
echo ""
echo ""

echo "=========================================="
echo "  ✅ Pruebas completadas"
echo "=========================================="
