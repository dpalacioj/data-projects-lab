"""Configuracion centralizada. Un solo lugar para rutas, credenciales y parametros.

Las credenciales NUNCA van hardcodeadas. Se leen de variables de entorno (o del
archivo `.env` cargado por `python-dotenv`). Este es el patron de produccion.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"      # JSON crudos (gitignored)
DB_DIR = DATA_DIR / "db"        # Archivo SQLite (gitignored)
REPORTS_DIR = PROJECT_DIR / "reports"

for d in (RAW_DIR, DB_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Carga de variables de entorno (.env en la raiz del proyecto)
# ---------------------------------------------------------------------------
# `override=False` -> si la variable ya esta en el entorno (ej. en CI), gana.
# El `.env.example` documenta cada variable y queda commiteado.
ENV_PATH = PROJECT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)


def build_database_url() -> str:
    """Devuelve la URL de conexion a la base.

    Prioridad:

    1. Si existe `DATABASE_URL` en el entorno, se usa tal cual. Esto es lo
       normal: la URL se inyecta desde fuera (12-factor app).
    2. Si no existe pero hay credenciales de Postgres, la arma a partir de
       `DB_USER/DB_PASSWORD/DB_HOST/DB_PORT/DB_NAME`.
    3. Si tampoco, cae al SQLite por defecto en `data/db/eventos.db`.

    El paso 2 ilustra como se arma una URL en produccion sin filtrar la
    password en el codigo ni en git.
    """
    url = os.getenv("DATABASE_URL")
    if url:
        return url

    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME")

    if all([user, password, host, name]):
        return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{name}"

    default_sqlite = DB_DIR / "eventos.db"
    return f"sqlite:///{default_sqlite}"


DATABASE_URL: str = build_database_url()


def database_url_safe(url: str = DATABASE_URL) -> str:
    """Oculta la password para loguear la URL sin filtrar secretos.

    Devuelve `postgresql+psycopg://user:***@host:5432/db`. Para SQLite,
    devuelve la URL tal cual (no tiene password).
    """
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    userinfo, hostinfo = rest.split("@", 1)
    if ":" in userinfo:
        user, _pwd = userinfo.split(":", 1)
        userinfo = f"{user}:***"
    return f"{scheme}://{userinfo}@{hostinfo}"


# ---------------------------------------------------------------------------
# Parametros del generador sintetico (todos configurables por entorno)
# ---------------------------------------------------------------------------
DIAS = int(os.getenv("EVENTOS_DIAS", "60"))
PEDIDOS_POR_DIA = int(os.getenv("EVENTOS_PEDIDOS_POR_DIA", "1700"))
N_CLIENTES = int(os.getenv("EVENTOS_CLIENTES", "20000"))
N_PRODUCTOS = int(os.getenv("EVENTOS_PRODUCTOS", "500"))
SEMILLA = int(os.getenv("EVENTOS_SEMILLA", "42"))

# ---------------------------------------------------------------------------
# Parametros de carga (batch size al insertar en la base)
# ---------------------------------------------------------------------------
# SQLite rinde bien con batches de ~5-10K filas. Postgres acepta mas.
BATCH_SIZE = int(os.getenv("INGESTA_BATCH_SIZE", "5000"))

# Nombres de tablas (en un solo sitio, por si cambian de convencion).
TABLA_CLIENTES = "clientes"
TABLA_PRODUCTOS = "productos"
TABLA_PEDIDOS = "pedidos"
TABLA_ITEMS = "items_pedido"
