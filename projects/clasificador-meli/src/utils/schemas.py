# app/src/api/schemas.py

from typing import List, Optional, Union
from pydantic import BaseModel


# --- Anidados ---
class Country(BaseModel):
    name: Optional[str]
    id: Optional[str]

class State(BaseModel):
    name: Optional[str]
    id: Optional[str]

class City(BaseModel):
    name: Optional[str]
    id: Optional[str]

class SellerAddress(BaseModel):
    country: Optional[Country]
    state: Optional[State]
    city: Optional[City]

class Shipping(BaseModel):
    local_pick_up: Optional[bool]
    free_shipping: Optional[bool]
    mode: Optional[str]
    tags: Optional[List[str]]

class PaymentMethod(BaseModel):
    description: str
    id: Optional[str]
    type: Optional[str]


# --- Principal ---
class ItemFullSchema(BaseModel):
    seller_address: Optional[SellerAddress]
    warranty: Optional[str]
    price: float
    shipping: Optional[Shipping]
    non_mercado_pago_payment_methods: Optional[List[PaymentMethod]]
    seller_id: Optional[int]
    listing_type_id: Optional[str]
    available_quantity: int
    sold_quantity: float
    start_time: Optional[Union[str, int]]
    stop_time: Optional[Union[str, int]]
    accepts_mercadopago: Optional[bool]
    currency_id: Optional[str]
    tags: Optional[List[str]]
    automatic_relist: Optional[bool]
    category_id: Optional[str]
    parent_item_id: Optional[str]
    status: Optional[str]
    video_id: Optional[str]
    original_price: Optional[float]
    buying_mode: Optional[str]
    initial_quantity: int


# Para múltiples entradas
class ItemsFullSchema(BaseModel):
    inputs: List[ItemFullSchema]
