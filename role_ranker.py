"""
role_ranker.py
--------------
Stage III: Hybrid Role-Relevance Ranking Engine.

Re-ranks the single shared pool of insights for 4 target roles:
1. Business Owner
2. Store Manager
3. Marketing Head
4. Inventory Manager

Formula:
  Hybrid_Score = alpha * LLM_Score + (1 - alpha) * Rule_Score
"""

from typing import Dict, List, Any, Tuple, Optional


# 4 Target Role Definitions & Domain Field Preference Weights
ROLE_PROFILES: Dict[str, Dict[str, Any]] = {
    "Business Owner": {
        "description": "C-suite / Business Owner seeking macro performance, overall revenue, profit margins, and growth trends.",
        "preferred_categories": ["efficiency", "trend", "attribution"],
        "field_weights": {
            "sales_amount": 1.0,
            "profit": 1.0,
            "margin_pct": 0.95,
            "growth_pct": 0.90,
            "date": 0.70,
            "category": 0.60,
            "region": 0.50
        },
        "llm_prompt_persona": "You are a Business Owner / Chief Executive Officer evaluating strategic financial health, top-line revenue growth, and profit margins."
    },
    "Store Manager": {
        "description": "Retail Store / Operations Manager focused on regional performance, store returns, transaction volume, and operational hiccups.",
        "preferred_categories": ["outlier", "efficiency", "attribution"],
        "field_weights": {
            "region": 1.0,
            "store_id": 1.0,
            "return_status": 0.95,
            "order_id": 0.85,
            "sales_amount": 0.80,
            "units_sold": 0.75
        },
        "llm_prompt_persona": "You are a Regional Retail Store Manager evaluating operational throughput, regional sales variations, return rate spikes, and order fulfillment."
    },
    "Marketing Head": {
        "description": "CMO / Marketing Head tracking category trends, promotional discount impact, customer segment conversions, and campaign efficiency.",
        "preferred_categories": ["attribution", "correlation", "trend"],
        "field_weights": {
            "category": 1.0,
            "sub_category": 0.95,
            "discount_pct": 0.95,
            "conversion_rate": 0.90,
            "campaign_id": 0.90,
            "customer_segment": 0.85,
            "units_sold": 0.70
        },
        "llm_prompt_persona": "You are a Marketing Head / CMO assessing brand campaign effectiveness, promotional elasticity, customer segment behavior, and department revenue contribution."
    },
    "Inventory Manager": {
        "description": "Warehouse & Supply Chain Lead overseeing stock levels, SKU turnover, returns, and inventory risk alerts.",
        "preferred_categories": ["efficiency", "outlier", "correlation"],
        "field_weights": {
            "inventory_level": 1.0,
            "units_sold": 0.95,
            "product_name": 0.90,
            "return_status": 0.85,
            "category": 0.75,
            "sub_category": 0.70
        },
        "llm_prompt_persona": "You are a Supply Chain & Inventory Manager tracking stockout risk, warehouse turnover, SKU demand, and return volume."
    }
}


class RoleRanker:
    """
    Reranks the single shared insight pool dynamically for any requested role.
    Computes deterministic Rule Scores and simulates/calls LLM Relevance Scores.
    """

    def __init__(self, alpha: float = 0.60):
        self.alpha = max(0.0, min(1.0, alpha))
        self.role_profiles = ROLE_PROFILES

    def calculate_rule_score(self, insight: Dict[str, Any], role_name: str) -> float:
        """
        Calculates a deterministic rule-based relevance score (0.0 to 1.0)
        based on column field matches and category preferences.
        """
        profile = self.role_profiles.get(role_name)
        if not profile:
            return 0.50

        field_weights = profile["field_weights"]
        preferred_cats = profile["preferred_categories"]

        # Category Match Score
        cat_score = 0.80 if insight.get("category") in preferred_cats else 0.40

        # Field Match Score
        assoc_cols = insight.get("associated_columns", [])
        if assoc_cols:
            col_scores = [field_weights.get(col, 0.20) for col in assoc_cols]
            field_score = sum(col_scores) / len(col_scores)
        else:
            field_score = 0.30

        # Combine
        rule_score = 0.60 * field_score + 0.40 * cat_score
        return round(min(1.0, max(0.0, rule_score)), 3)

    def calculate_llm_score(self, insight: Dict[str, Any], role_name: str) -> float:
        """
        Simulates / executes semantic LLM relevance scoring (0.0 to 1.0).
        Evaluates how critical the insight title & explanation are to the role persona.
        """
        profile = self.role_profiles.get(role_name, {})
        persona = profile.get("llm_prompt_persona", "")
        title = insight.get("title", "")
        explanation = insight.get("simple_explanation", "")

        # Heuristic semantic evaluator (simulating LLM prompt response)
        score = 0.50

        if role_name == "Business Owner":
            if any(term in title.lower() or term in explanation.lower() for term in ["revenue", "profit", "aov", "grew", "total", "growth"]):
                score += 0.40
            if "spike" in title.lower() or "spike" in explanation.lower():
                score += 0.20

        elif role_name == "Store Manager":
            if any(term in title.lower() or term in explanation.lower() for term in ["region", "return", "spike", "order", "store"]):
                score += 0.45
            if "outlier" in insight.get("category", ""):
                score += 0.25

        elif role_name == "Marketing Head":
            if any(term in title.lower() or term in explanation.lower() for term in ["category", "discount", "dominates", "volume", "link"]):
                score += 0.45
            if "correlation" in insight.get("category", "") or "attribution" in insight.get("category", ""):
                score += 0.25

        elif role_name == "Inventory Manager":
            if any(term in title.lower() or term in explanation.lower() for term in ["stock", "return", "volume", "units", "sku"]):
                score += 0.45
            if "return" in title.lower() or "stock" in title.lower():
                score += 0.30

        return round(min(1.0, max(0.10, score)), 3)

    def rank_insights_for_role(
        self,
        insight_pool: List[Dict[str, Any]],
        role_name: str,
        override_alpha: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Re-ranks the shared insight pool for a specific role using hybrid scoring.
        
        Returns sorted list of insights enriched with ranking metadata:
          - hybrid_score
          - llm_score
          - rule_score
          - role_rank
        """
        alpha = self.alpha if override_alpha is None else max(0.0, min(1.0, override_alpha))
        ranked_pool = []

        for ins in insight_pool:
            ins_copy = dict(ins)
            rule_score = self.calculate_rule_score(ins, role_name)
            llm_score = self.calculate_llm_score(ins, role_name)
            hybrid_score = (alpha * llm_score) + ((1.0 - alpha) * rule_score)

            ins_copy["rule_score"] = rule_score
            ins_copy["llm_score"] = llm_score
            ins_copy["hybrid_score"] = round(hybrid_score, 3)
            ins_copy["target_role"] = role_name
            ranked_pool.append(ins_copy)

        # Sort descending by hybrid_score
        ranked_pool.sort(key=lambda x: x["hybrid_score"], reverse=True)

        for rank, ins in enumerate(ranked_pool, start=1):
            ins["role_rank"] = rank

        return ranked_pool

    def rank_all_roles(
        self,
        insight_pool: List[Dict[str, Any]],
        override_alpha: Optional[float] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generates ranked insight lists for all 4 roles simultaneously."""
        results = {}
        for role_name in self.role_profiles.keys():
            results[role_name] = self.rank_insights_for_role(insight_pool, role_name, override_alpha)
        return results


if __name__ == "__main__":
    from insight_generator import InsightGenerator
    import pandas as pd

    test_df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=10, freq="D"),
        "sales_amount": [100, 120, 150, 90, 80, 500, 110, 130, 140, 160],
        "units_sold": [1, 2, 3, 1, 1, 10, 2, 2, 3, 4],
        "discount_pct": [0, 5, 10, 0, 0, 25, 5, 5, 10, 15],
        "category": ["Apparel", "Apparel", "Electronics", "Electronics", "Apparel", "Electronics", "Apparel", "Electronics", "Apparel", "Electronics"],
        "region": ["West", "East", "West", "East", "South", "West", "South", "Central", "West", "East"],
        "return_status": ["No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "Yes"]
    })

    gen = InsightGenerator(test_df)
    pool = gen.generate_all_insights()

    ranker = RoleRanker(alpha=0.60)
    all_ranked = ranker.rank_all_roles(pool)

    print("--- RE-RANKING RESULTS PER ROLE ---")
    for role, ranked in all_ranked.items():
        print(f"\nRole: [{role}] Top Insight:")
        top = ranked[0]
        print(f"  Rank #1: [{top['insight_id']}] {top['title']}")
        print(f"  Hybrid Score: {top['hybrid_score']} (LLM: {top['llm_score']}, Rule: {top['rule_score']})")
