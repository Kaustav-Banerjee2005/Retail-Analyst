"""
schema_normalizer-v2.py
-----------------------
Stage I: Schema Normalization for Retail/Marketing Analytics Framework (Version 2 - Bipartite Matcher).

Maps raw, heterogeneous column names from arbitrary retail/marketing datasets
into a fixed, standardized marketing vocabulary. Uses global bipartite matching,
exact alias dictionaries, token boundary matching, and fuzzy semantic similarity
to handle both standard retail columns and unseen column names gracefully without
order-dependent greedy mis-assignments.
"""

import re
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from difflib import SequenceMatcher


# Standardized Marketing & Retail Target Vocabulary
STANDARD_MARKETING_VOCABULARY: Dict[str, Dict[str, Any]] = {
    "order_id": {
        "description": "Unique transaction or order identifier",
        "aliases": ["order_id", "order_no", "order_number", "transaction_id", "invoice_no", "receipt_id", "order id", "inv_id", "order_num"],
        "data_type": "categorical"
    },
    "date": {
        "description": "Transaction or event date/time",
        "aliases": ["date", "order_date", "trans_date", "timestamp", "sale_date", "day", "created_at", "purchase_date", "trans_dt"],
        "data_type": "datetime"
    },
    "ship_date": {
        "description": "Shipping or dispatch date",
        "aliases": ["ship_date", "shipping_date", "dispatch_date", "ship_dt"],
        "data_type": "datetime"
    },
    "ship_mode": {
        "description": "Shipping method or fulfillment mode",
        "aliases": ["ship_mode", "shipping_mode", "ship_method", "fulfillment_mode"],
        "data_type": "categorical"
    },
    "sales_amount": {
        "description": "Total gross revenue or transaction sales value ($)",
        "aliases": ["sales", "sales_amount", "revenue", "gross_sales", "total_amount", "order_value", "turnover", "sales_usd", "gross_sales_usd"],
        "data_type": "numeric"
    },
    "units_sold": {
        "description": "Quantity or volume of items purchased",
        "aliases": ["quantity", "units", "units_sold", "qty", "item_count", "volume", "units_purchased", "item_qty"],
        "data_type": "numeric"
    },
    "unit_price": {
        "description": "Selling price per item unit ($)",
        "aliases": ["unit_price", "price", "item_price", "mrp", "list_price", "selling_price"],
        "data_type": "numeric"
    },
    "discount_pct": {
        "description": "Discount or promotional markdown rate (% or $)",
        "aliases": ["discount", "discount_pct", "discount_rate", "promo_pct", "markdown", "discount_percentage", "promo_discount"],
        "data_type": "numeric"
    },
    "profit": {
        "description": "Gross profit or net contribution margin ($)",
        "aliases": ["profit", "net_profit", "margin", "gross_margin", "profit_usd", "earnings"],
        "data_type": "numeric"
    },
    "cost_price": {
        "description": "Cost of goods sold (COGS) or acquisition cost per unit ($)",
        "aliases": ["cost", "cogs", "unit_cost", "acquisition_cost", "product_cost", "cost_price"],
        "data_type": "numeric"
    },
    "category": {
        "description": "High-level product category",
        "aliases": ["category", "prod_category", "department", "dept", "product_type", "macro_category"],
        "data_type": "categorical"
    },
    "sub_category": {
        "description": "Granular product sub-category or line",
        "aliases": ["sub_category", "subcategory", "sub_cat", "product_group", "product_line"],
        "data_type": "categorical"
    },
    "product_name": {
        "description": "Product SKU, title, or item description",
        "aliases": ["product_name", "product", "item_name", "title", "description", "item_desc", "product_title"],
        "data_type": "categorical"
    },
    "product_id": {
        "description": "Product unique SKU or identifier",
        "aliases": ["product_id", "prod_id", "item_id", "sku", "sku_id"],
        "data_type": "categorical"
    },
    "customer_id": {
        "description": "Unique customer or user identifier",
        "aliases": ["customer_id", "cust_id", "user_id", "client_id", "shopper_id", "customer_no"],
        "data_type": "categorical"
    },
    "customer_name": {
        "description": "Customer full name or client label",
        "aliases": ["customer_name", "cust_name", "client_name", "shopper_name", "customer"],
        "data_type": "categorical"
    },
    "customer_segment": {
        "description": "Customer demographic or tier segment",
        "aliases": ["segment", "customer_segment", "cust_type", "tier", "loyalty_group", "target_group", "customer_loyalty_tier"],
        "data_type": "categorical"
    },
    "country": {
        "description": "Country or nation",
        "aliases": ["country", "nation", "country_code"],
        "data_type": "categorical"
    },
    "city": {
        "description": "City or town",
        "aliases": ["city", "town", "municipality"],
        "data_type": "categorical"
    },
    "state": {
        "description": "State or province",
        "aliases": ["state", "province", "territory_state"],
        "data_type": "categorical"
    },
    "postal_code": {
        "description": "Postal code or ZIP code",
        "aliases": ["postal_code", "zip", "zip_code", "postal", "postcode"],
        "data_type": "categorical"
    },
    "region": {
        "description": "Geographic territory, region, or market zone",
        "aliases": ["region", "territory", "market", "zone", "geography", "location", "region_zone"],
        "data_type": "categorical"
    },
    "store_id": {
        "description": "Retail store outlet identifier or store channel",
        "aliases": ["store_id", "store", "outlet", "branch", "channel", "store_name", "fulfillment_center"],
        "data_type": "categorical"
    },
    "return_status": {
        "description": "Product return or order refund indicator",
        "aliases": ["returned", "return_status", "is_returned", "refund_flag", "return_flag", "status"],
        "data_type": "categorical"
    },
    "inventory_level": {
        "description": "Stock volume on hand or warehouse inventory level",
        "aliases": ["inventory", "inventory_level", "stock", "stock_qty", "warehouse_stock", "stock_on_hand"],
        "data_type": "numeric"
    },
    "marketing_spend": {
        "description": "Advertising campaign expenditure ($)",
        "aliases": ["marketing_spend", "ad_spend", "campaign_cost", "marketing_cost", "spend", "ad_budget"],
        "data_type": "numeric"
    },
    "campaign_id": {
        "description": "Marketing campaign identifier or promo code",
        "aliases": ["campaign", "campaign_id", "promo_code", "campaign_name", "ad_group"],
        "data_type": "categorical"
    },
    "conversion_rate": {
        "description": "Campaign click-through or customer conversion rate (%)",
        "aliases": ["conversion_rate", "cvr", "ctr", "conversion_pct", "conversion"],
        "data_type": "numeric"
    }
}


def clean_column_name(col: str) -> str:
    """Normalize raw string for string matching."""
    col = str(col).lower().strip()
    col = re.sub(r'[\s\-_]+', '_', col)
    col = re.sub(r'[^a-z0-9_]', '', col)
    return col


def calculate_similarity(s1: str, s2: str) -> float:
    """Compute string similarity using SequenceMatcher."""
    return SequenceMatcher(None, s1, s2).ratio()


class SchemaNormalizer:
    """
    Normalizes arbitrary marketing CSV column names into standard vocabulary terms.
    Uses global bipartite scoring to eliminate column misallocation order biases.
    """

    def __init__(self, vocab: Optional[Dict[str, Dict[str, Any]]] = None):
        self.vocab = vocab or STANDARD_MARKETING_VOCABULARY

    def map_columns(self, df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
        """
        Maps dataframe column names to standard vocabulary terms.
        
        Returns:
            mapping: {raw_column_name: standard_vocab_term}
            details: {raw_column_name: {"standard_term": ..., "confidence": ..., "method": ..., "type": ...}}
        """
        candidates = []  # List of tuples: (score, raw_col, std_term, method)

        for raw_col in df.columns:
            cleaned = clean_column_name(raw_col)

            for std_term, info in self.vocab.items():
                aliases = [clean_column_name(a) for a in info["aliases"]] + [clean_column_name(std_term)]
                
                # Step 1: Direct Exact Alias Match
                if cleaned in aliases:
                    candidates.append((1.00, raw_col, std_term, "exact_alias"))
                    continue

                # Step 2: Token Boundary Match / Token Overlap
                for alias in aliases:
                    if len(alias) >= 3 and len(cleaned) >= 3:
                        cleaned_tokens = set(cleaned.split("_"))
                        alias_tokens = set(alias.split("_"))
                        if alias in cleaned_tokens or cleaned in alias_tokens or alias == cleaned:
                            candidates.append((0.90, raw_col, std_term, "token_match"))
                        elif alias in cleaned or cleaned in alias:
                            len_ratio = min(len(alias), len(cleaned)) / max(len(alias), len(cleaned))
                            if len_ratio >= 0.50:
                                candidates.append((round(0.75 * len_ratio, 2), raw_col, std_term, "substring"))

                # Step 3: Fuzzy Similarity
                for alias in aliases:
                    sim = calculate_similarity(cleaned, alias)
                    if sim >= 0.70:
                        candidates.append((round(sim * 0.85, 2), raw_col, std_term, "fuzzy_similarity"))

        # Sort candidate matches by confidence score descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        mapping: Dict[str, str] = {}
        details: Dict[str, Dict[str, Any]] = {}
        assigned_cols = set()
        used_std_terms = set()

        # Global optimal match assignment
        for score, raw_col, std_term, method in candidates:
            if raw_col in assigned_cols:
                continue
            if std_term in used_std_terms:
                continue

            mapping[raw_col] = std_term
            details[raw_col] = {
                "standard_term": std_term,
                "confidence": score,
                "method": method,
                "inferred_dtype": str(df[raw_col].dtype)
            }
            assigned_cols.add(raw_col)
            used_std_terms.add(std_term)

        # Fallback for completely unseen/unassigned columns
        for raw_col in df.columns:
            if raw_col not in assigned_cols:
                cleaned = clean_column_name(raw_col)
                fallback_term = f"custom_{cleaned}" if cleaned else "custom_column"
                mapping[raw_col] = fallback_term
                details[raw_col] = {
                    "standard_term": fallback_term,
                    "confidence": 0.50,
                    "method": "custom_unseen_fallback",
                    "inferred_dtype": str(df[raw_col].dtype)
                }

        return mapping, details

    def normalize_dataframe(self, df: pd.DataFrame, custom_mapping: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Applies mapping to rename DataFrame columns to normalized standard terms.
        Allows interactive custom_mapping overrides.
        """
        if custom_mapping is None:
            mapping, _ = self.map_columns(df)
        else:
            mapping = custom_mapping

        normalized_df = df.rename(columns=mapping)
        return normalized_df, mapping


if __name__ == "__main__":
    test_data = pd.DataFrame({
        "Trans_Date": ["2023-01-01", "2023-01-02"],
        "Gross_Sales_USD": [150.0, 200.0],
        "Item_Qty": [2, 5],
        "Promo_Discount_%": [10, 15],
        "Region_Zone": ["West", "East"],
        "Customer Name": ["John Doe", "Jane Smith"],
        "State": ["California", "New York"]
    })
    
    normalizer = SchemaNormalizer()
    mapping, details = normalizer.map_columns(test_data)
    print("Mapping Result:")
    for k, v in details.items():
        print(f"  '{k}' -> '{v['standard_term']}' (confidence: {v['confidence']}, method: {v['method']})")
    
    norm_df, _ = normalizer.normalize_dataframe(test_data)
    print("\nNormalized Columns:", list(norm_df.columns))
