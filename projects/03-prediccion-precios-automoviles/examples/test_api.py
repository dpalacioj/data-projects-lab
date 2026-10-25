"""
Script de Prueba para la API de Predicción de Precios

Este script demuestra cómo hacer requests a la API usando Python.

Requisitos:
    pip install requests

Uso:
    1. Inicia la API: python app_fastapi.py
    2. En otra terminal: python examples/test_api.py
"""

import requests
import json
from typing import Dict, Any


# URL base de la API
API_URL = "http://localhost:8000"


def print_response(title: str, response: requests.Response):
    """Imprime la respuesta de forma bonita"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"Status Code: {response.status_code}")

    try:
        data = response.json()
        print(f"Response:\n{json.dumps(data, indent=2, ensure_ascii=False)}")
    except:
        print(f"Response: {response.text}")

    print(f"{'='*70}\n")


def test_health_check():
    """Prueba 1: Health Check"""
    response = requests.get(f"{API_URL}/health")
    print_response("1️⃣  Health Check", response)
    return response.status_code == 200


def test_model_info():
    """Prueba 2: Información del Modelo"""
    response = requests.get(f"{API_URL}/model-info")
    print_response("2️⃣  Información del Modelo", response)
    return response.status_code == 200


def test_prediction(data: Dict[str, Any], title: str):
    """Realiza una predicción"""
    response = requests.post(
        f"{API_URL}/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    print_response(title, response)
    return response.status_code == 200


def main():
    """Ejecuta todas las pruebas"""
    print("\n" + "="*70)
    print("  🚀 Probando API de Predicción de Precios de Automóviles")
    print("="*70)

    # Verificar que la API está corriendo
    try:
        requests.get(f"{API_URL}/health", timeout=2)
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: La API no está corriendo")
        print("\nInicia la API con:")
        print("  python app_fastapi.py")
        print("\nLuego ejecuta este script nuevamente.")
        return

    # Ejecutar pruebas
    results = []

    # Test 1: Health Check
    results.append(test_health_check())

    # Test 2: Model Info
    results.append(test_model_info())

    # Test 3: Toyota SUV 2020 (Auto estándar)
    toyota_data = {
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
    results.append(test_prediction(toyota_data, "3️⃣  Predicción: Toyota SUV 2020"))

    # Test 4: BMW Sedán 2022 (Auto de lujo)
    bmw_data = {
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
    }
    results.append(test_prediction(bmw_data, "4️⃣  Predicción: BMW Sedán 2022 (Lujo)"))

    # Test 5: Honda Hatchback 2018 (Auto económico)
    honda_data = {
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
    }
    results.append(test_prediction(honda_data, "5️⃣  Predicción: Honda Hatchback 2018 (Económico)"))

    # Test 6: Mercedes-Benz SUV 2023 (Auto premium)
    mercedes_data = {
        "marca": "Mercedes-Benz",
        "tipo_carroceria": "SUV",
        "año": 2023,
        "kilometraje": 10000,
        "tipo_combustible": "Eléctrico",
        "transmision": "Automática",
        "cilindrada": 2000,
        "potencia": 300,
        "peso": 2000,
        "consumo": 0.0,
        "color": "Plata",
        "edad_propietarios": 1,
        "calificacion_estado": 10.0,
        "region_venta": "Centro"
    }
    results.append(test_prediction(mercedes_data, "6️⃣  Predicción: Mercedes-Benz SUV 2023 (Premium)"))

    # Test 7: Ford Pickup 2015 (Auto de trabajo)
    ford_data = {
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
    }
    results.append(test_prediction(ford_data, "7️⃣  Predicción: Ford Pickup 2015 (Trabajo)"))

    # Resumen
    print("="*70)
    print("  📊 Resumen de Pruebas")
    print("="*70)
    print(f"Total de pruebas: {len(results)}")
    print(f"✅ Exitosas: {sum(results)}")
    print(f"❌ Fallidas: {len(results) - sum(results)}")
    print("="*70)

    if all(results):
        print("\n🎉 ¡Todas las pruebas pasaron exitosamente!")
    else:
        print("\n⚠️  Algunas pruebas fallaron. Revisa los errores arriba.")


if __name__ == "__main__":
    main()
