
import logging
from typing import List, Dict, Any, Optional
from supabase_agent import supabase_agent

logger = logging.getLogger("ecommerce_db")




def init_db():
    # Tables are assumed to be managed in Supabase.
    pass

def search_products(keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
    try:
        return supabase_agent.search_like_any(
            table="products",
            columns=["name", "description"],
            keyword=keyword,
            limit=limit,
        )
    except Exception as e:
        logger.error(f"Error searching products in Supabase: {e}")
        return []

def get_product_by_id(product_id: int) -> Optional[Dict[str, Any]]:
    try:
        return supabase_agent.read_one('products', filters={'id': product_id})
    except Exception as e:
        logger.error(f"Error getting product by id in Supabase: {e}")
    return None

def place_order(customer_name: str, product_id: int, quantity: int = 1) -> Dict[str, Any]:
    prod = get_product_by_id(product_id)
    if not prod:
        return {"error": "product_not_found"}
    if prod["stock"] < quantity:
        return {"error": "insufficient_stock", "available": prod["stock"]}
    try:
        # Decrement stock
        new_stock = prod["stock"] - quantity
        supabase_agent.update('products', filters={'id': product_id}, updates={"stock": new_stock})
        # Insert order
        order_data = {
            "customer_name": customer_name,
            "product_id": product_id,
            "quantity": quantity,
            "status": "processing"
        }
        response = supabase_agent.create('orders', order_data)
        if response:
            order_id = response[0].get("order_id") or response[0].get("id")
            return {"order_id": order_id, "status": "processing"}
    except Exception as e:
        logger.error(f"Error placing order in Supabase: {e}")
    return {"error": "failed_to_place_order"}

def seed_sample_products():
    try:
        existing = supabase_agent.read('products', limit=1)
        if existing:
            return # Already seeded
        sample = [
            {"name": "Samsung Galaxy A55", "category": "Electronics", "description": "6.6 inch AMOLED, 8GB RAM", "price": 35000, "stock": 20},
            {"name": "Dell Inspiron 15", "category": "Laptop", "description": "15.6 inch FHD, Intel i5", "price": 75000, "stock": 7},
            {"name": "Yoga Mat 6mm", "category": "Sports", "description": "Non-slip TPE yoga mat", "price": 900, "stock": 50},
        ]
        supabase_agent.create_many('products', sample)
        logger.info("Sample products seeded to Supabase.")
    except Exception as e:
        logger.error(f"Error seeding products in Supabase: {e}")
