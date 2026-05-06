"""
Order Management Tools for the Bengali E-commerce Voice Agent.

Provides:
  - place_order        : Create a new order and decrement inventory
  - cancel_order       : Cancel an order by ID or order number and restore stock
  - get_order_status   : Check the status of an order
  - list_customer_orders : List all orders for a customer (by name/phone)
  - update_order_status  : Update any field on an order (admin use)
"""

import logging
from typing import Optional
from supabase_agent import SupabaseAgent

logger = logging.getLogger("order_tools")

# ─── helpers ────────────────────────────────────────────────────────────────

def _get_sa() -> SupabaseAgent:
    return SupabaseAgent()


def _resolve_product(sa: SupabaseAgent, product_name: str) -> Optional[dict]:
    """Find a product by (partial) name — case-insensitive."""
    results = sa.search_like_any("products", ["name"], product_name, limit=5)
    if not results:
        return None
    # Prefer exact match first, otherwise return first fuzzy match
    name_lower = product_name.lower()
    for p in results:
        if p.get("name", "").lower() == name_lower:
            return p
    return results[0]


# ─── public API ─────────────────────────────────────────────────────────────

def place_order(
    customer_name: str,
    product_name: str,
    quantity: int = 1,
    customer_phone: Optional[str] = None,
    notes: Optional[str] = None,
) -> dict:
    """
    Place a new order.

    1. Resolve product by name (fuzzy search).
    2. Check inventory.
    3. Insert order row.
    4. Decrement inventory_count on products table.

    Returns:
        dict with order_id, status, total_price  OR  error key.
    """
    sa = _get_sa()

    # 1. Find product
    product = _resolve_product(sa, product_name)
    if not product:
        return {"error": "product_not_found", "message": f"পণ্য '{product_name}' পাওয়া যায়নি।"}

    # 2. Check stock
    stock = product.get("inventory_count", 0) or 0
    if stock < quantity:
        return {
            "error": "insufficient_stock",
            "available": stock,
            "message": f"স্টকে মাত্র {stock}টি পণ্য আছে।",
        }

    # 3. Determine price
    unit_price = float(product.get("discount_price") or product.get("base_price") or 0)
    total_price = unit_price * quantity

    # 4. Insert order
    order_data = {
        "customer_name": customer_name,
        "customer_phone": customer_phone or "",
        "product_id": product.get("id"),
        "product_name": product.get("name"),
        "quantity": quantity,
        "unit_price": unit_price,
        "total_price": total_price,
        "status": "confirmed",
        "notes": notes or "",
    }
    try:
        result = sa.create("orders", order_data)
        if not result:
            return {"error": "db_insert_failed", "message": "অর্ডার সেভ করতে ব্যর্থ হয়েছে।"}
        order_id = result[0].get("id", "?")

        # 5. Decrement stock
        sa.update(
            "products",
            filters={"id": product["id"]},
            updates={"inventory_count": stock - quantity},
        )

        logger.info(f"Order placed: {order_id} | {customer_name} | {product['name']} x{quantity} = ৳{total_price}")
        return {
            "order_id": order_id,
            "status": "confirmed",
            "product_name": product["name"],
            "quantity": quantity,
            "unit_price": unit_price,
            "total_price": total_price,
            "message": (
                f"অর্ডার সফলভাবে নেওয়া হয়েছে! "
                f"অর্ডার আইডি: {str(order_id)[:8]}। "
                f"{product['name']} × {quantity} = ৳{total_price:.0f}।"
            ),
        }
    except Exception as e:
        logger.error(f"place_order error: {e}")
        return {"error": str(e), "message": "অর্ডার দিতে সমস্যা হয়েছে।"}


def cancel_order(order_id: str, reason: Optional[str] = None) -> dict:
    """
    Cancel an existing order by its ID and restore stock.

    Returns:
        dict with success message OR error key.
    """
    sa = _get_sa()
    try:
        # Fetch order
        order = sa.read_one("orders", {"id": order_id})
        if not order:
            return {"error": "order_not_found", "message": f"অর্ডার আইডি '{order_id}' পাওয়া যায়নি।"}

        current_status = order.get("status", "")
        if current_status in ("cancelled", "delivered"):
            return {
                "error": "cannot_cancel",
                "message": f"এই অর্ডারটি ইতিমধ্যে '{current_status}' অবস্থায় আছে, বাতিল করা যাবে না।",
            }

        # Update status
        updates = {"status": "cancelled"}
        if reason:
            existing_notes = order.get("notes", "") or ""
            updates["notes"] = f"{existing_notes}\nCancellation reason: {reason}".strip()

        sa.update("orders", filters={"id": order_id}, updates=updates)

        # Restore stock
        product_id = order.get("product_id")
        quantity = order.get("quantity", 0)
        if product_id and quantity:
            product = sa.read_one("products", {"id": product_id})
            if product:
                current_stock = product.get("inventory_count", 0) or 0
                sa.update(
                    "products",
                    filters={"id": product_id},
                    updates={"inventory_count": current_stock + quantity},
                )

        logger.info(f"Order cancelled: {order_id}")
        return {
            "order_id": order_id,
            "status": "cancelled",
            "message": f"অর্ডার {str(order_id)[:8]} সফলভাবে বাতিল করা হয়েছে। স্টক পুনরুদ্ধার করা হয়েছে।",
        }
    except Exception as e:
        logger.error(f"cancel_order error: {e}")
        return {"error": str(e), "message": "অর্ডার বাতিল করতে সমস্যা হয়েছে।"}


def get_order_status(order_id: str) -> dict:
    """
    Get the current status and details of an order.
    """
    sa = _get_sa()
    try:
        order = sa.read_one("orders", {"id": order_id})
        if not order:
            return {"error": "order_not_found", "message": f"অর্ডার '{order_id}' পাওয়া যায়নি।"}

        status = order.get("status", "unknown")
        status_bangla = {
            "pending":    "অপেক্ষমান",
            "confirmed":  "নিশ্চিত",
            "processing": "প্রক্রিয়াধীন",
            "shipped":    "পাঠানো হয়েছে",
            "delivered":  "ডেলিভারি হয়েছে",
            "cancelled":  "বাতিল",
        }.get(status, status)

        return {
            "order_id": order_id,
            "status": status,
            "status_bn": status_bangla,
            "product_name": order.get("product_name"),
            "quantity": order.get("quantity"),
            "total_price": order.get("total_price"),
            "customer_name": order.get("customer_name"),
            "created_at": order.get("created_at"),
            "message": (
                f"অর্ডার আইডি {str(order_id)[:8]} — "
                f"{order.get('product_name')} × {order.get('quantity')} — "
                f"বর্তমান অবস্থা: {status_bangla}।"
            ),
        }
    except Exception as e:
        logger.error(f"get_order_status error: {e}")
        return {"error": str(e), "message": "অর্ডারের তথ্য আনতে সমস্যা হয়েছে।"}


def list_customer_orders(customer_name: str = "", customer_phone: str = "", limit: int = 5) -> dict:
    """
    List all orders for a customer by name or phone number.
    """
    sa = _get_sa()
    try:
        orders = []
        if customer_phone:
            orders = sa.read("orders", filters={"customer_phone": customer_phone}, limit=limit)
        if not orders and customer_name:
            # Fuzzy match by name
            all_orders = sa.read("orders", limit=100)
            name_lower = customer_name.lower()
            orders = [o for o in all_orders if name_lower in o.get("customer_name", "").lower()][:limit]

        if not orders:
            return {
                "orders": [],
                "count": 0,
                "message": "কোনো অর্ডার পাওয়া যায়নি।",
            }

        summary = []
        for o in orders:
            status_bangla = {
                "pending": "অপেক্ষমান", "confirmed": "নিশ্চিত",
                "processing": "প্রক্রিয়াধীন", "shipped": "পাঠানো",
                "delivered": "ডেলিভারি হয়েছে", "cancelled": "বাতিল",
            }.get(o.get("status", ""), o.get("status", ""))
            summary.append(
                f"{str(o['id'])[:8]} — {o.get('product_name')} × {o.get('quantity')} "
                f"= ৳{o.get('total_price', 0):.0f} [{status_bangla}]"
            )

        return {
            "orders": orders,
            "count": len(orders),
            "message": f"{len(orders)}টি অর্ডার পাওয়া গেছে:\n" + "\n".join(summary),
        }
    except Exception as e:
        logger.error(f"list_customer_orders error: {e}")
        return {"error": str(e), "message": "অর্ডার তালিকা আনতে সমস্যা হয়েছে।"}


def update_order(order_id: str, updates: dict) -> dict:
    """
    Update any fields of an order (e.g. status, notes, quantity).
    Allowed fields: status, notes, quantity, customer_phone.
    """
    sa = _get_sa()
    ALLOWED_FIELDS = {"status", "notes", "quantity", "customer_phone", "customer_name"}
    safe_updates = {k: v for k, v in updates.items() if k in ALLOWED_FIELDS}

    if not safe_updates:
        return {"error": "no_valid_fields", "message": "আপডেট করার জন্য কোনো বৈধ ফিল্ড নেই।"}

    try:
        order = sa.read_one("orders", {"id": order_id})
        if not order:
            return {"error": "order_not_found", "message": f"অর্ডার '{order_id}' পাওয়া যায়নি।"}

        sa.update("orders", filters={"id": order_id}, updates=safe_updates)
        logger.info(f"Order updated: {order_id} -> {safe_updates}")
        return {
            "order_id": order_id,
            "updated_fields": list(safe_updates.keys()),
            "message": f"অর্ডার {str(order_id)[:8]} সফলভাবে আপডেট করা হয়েছে।",
        }
    except Exception as e:
        logger.error(f"update_order error: {e}")
        return {"error": str(e), "message": "অর্ডার আপডেট করতে সমস্যা হয়েছে।"}
