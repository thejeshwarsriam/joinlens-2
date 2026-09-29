"""Demo tables and preset queries that showcase common join fan-out mistakes."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


def sanitize_name(name: str) -> str:
    """Turn a file name into a safe SQL table name."""
    n = re.sub(r"\W+", "_", name).strip("_").lower() or "table"
    return f"t_{n}" if n[0].isdigit() else n


def make_demo_tables(seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_cust, n_orders = 200, 1000

    customers = pd.DataFrame(
        {
            "customer_id": np.arange(1, n_cust + 1),
            "region": rng.choice(["North", "South", "East", "West"], n_cust),
            "segment": rng.choice(["Retail", "SMB", "Enterprise"], n_cust),
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": np.arange(1, n_orders + 1),
            "customer_id": rng.integers(1, n_cust + 1, n_orders),
            "order_date": pd.to_datetime("2025-01-01")
            + pd.to_timedelta(rng.integers(0, 365, n_orders), unit="D"),
            "amount": rng.gamma(4, 60, n_orders).round(2),
        }
    )

    # ~35% of orders are paid in 2-3 instalments -> payments is one-to-many on order_id
    k = np.where(rng.random(n_orders) < 0.35, rng.integers(2, 4, n_orders), 1)
    payments = pd.DataFrame(
        {
            "payment_id": np.arange(1, k.sum() + 1),
            "order_id": np.repeat(orders["order_id"].to_numpy(), k),
        }
    )
    payments["paid_amount"] = rng.gamma(3, 40, len(payments)).round(2)
    payments["paid_at"] = pd.to_datetime("2025-01-05") + pd.to_timedelta(
        rng.integers(0, 365, len(payments)), unit="D"
    )

    # 1-5 line items per order
    m = rng.integers(1, 6, n_orders)
    order_items = pd.DataFrame(
        {
            "item_id": np.arange(1, m.sum() + 1),
            "order_id": np.repeat(orders["order_id"].to_numpy(), m),
        }
    )
    order_items["sku"] = rng.choice(["A100", "B200", "C300", "D400", "E500"], len(order_items))
    order_items["qty"] = rng.integers(1, 6, len(order_items))

    # one shipment for 80% of the orders
    shipped = orders.sample(frac=0.8, random_state=seed)
    shipments = pd.DataFrame(
        {
            "shipment_id": np.arange(1, len(shipped) + 1),
            "order_id": shipped["order_id"].to_numpy(),
            "customer_id": shipped["customer_id"].to_numpy(),
            "carrier": rng.choice(["DHL", "FedEx", "UPS"], len(shipped)),
        }
    )
    return {
        "customers": customers,
        "orders": orders,
        "payments": payments,
        "order_items": order_items,
        "shipments": shipments,
    }


PRESET_QUERIES: dict[str, str] = {
    "Revenue by region (fans out twice)": """SELECT c.region,
       SUM(o.amount)      AS revenue,
       SUM(p.paid_amount) AS paid,
       SUM(i.qty)         AS units,
       COUNT(*)           AS order_count
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN payments p ON o.order_id = p.order_id
LEFT JOIN order_items i ON o.order_id = i.order_id
GROUP BY c.region""",
    "Wrong grain: join on customer_id (many-to-many)": """SELECT COUNT(*) AS shipped_orders,
       SUM(o.amount) AS revenue
FROM orders o
JOIN shipments s ON o.customer_id = s.customer_id""",
    "Clean query (no fan-out)": """SELECT c.region, SUM(o.amount) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
GROUP BY c.region""",
}
