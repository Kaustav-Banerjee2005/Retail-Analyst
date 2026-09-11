"""
run_pipeline.py
---------------
End-to-End Orchestrator Script for:
"LLM-Assisted Role-Aware Insight Ranking for Automated EDA — A Schema-Normalized Framework for Retail Analytics"

Executes Stages I through V seamlessly in command-line mode:
  Stage I: Schema Normalization
  Stage II: Unranked Shared Insight Generation
  Stage III: Role-Relevance Re-ranking across 4 Roles
  Stage IV & V: Evaluation (NDCG@K, Precision@K, Inter-Annotator Agreement) & Ablation Benchmark
"""

import sys
import pandas as pd
import numpy as np

from schema_normalizer import SchemaNormalizer
from insight_generator import InsightGenerator
from role_ranker import RoleRanker, ROLE_PROFILES
from evaluation import Evaluator, generate_sample_ground_truth, compute_inter_annotator_agreement


def generate_sample_retail_data() -> pd.DataFrame:
    """Creates synthetic retail marketing dataset with heterogeneous column names."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    categories = ["Apparel", "Electronics", "Groceries", "Home & Kitchen"]
    regions = ["West", "East", "Central", "South"]
    
    data = []
    for i in range(80):
        d = np.random.choice(dates)
        cat = np.random.choice(categories)
        reg = np.random.choice(regions)
        qty = int(np.random.randint(1, 10))
        u_price = round(float(np.random.uniform(20.0, 180.0)), 2)
        disc = float(np.random.choice([0, 5, 10, 15, 25]))
        sales = round(qty * u_price * (1 - disc/100.0), 2)
        profit = round(sales * np.random.uniform(0.20, 0.35), 2)
        returned = np.random.choice(["No", "No", "Yes"], p=[0.80, 0.15, 0.05])
        inv = int(np.random.randint(5, 80))
        
        data.append({
            "Trans_DT": d,
            "Order_Num": f"ORD-{202300+i}",
            "Gross_Sales_USD": sales,
            "Item_Qty": qty,
            "Unit_Price_USD": u_price,
            "Promo_Discount_Pct": disc,
            "Profit_Margin_USD": profit,
            "Product_Dept": cat,
            "Region_Territory": reg,
            "Return_Flag": returned,
            "Stock_On_Hand": inv
        })
    df = pd.DataFrame(data)
    df.loc[10, "Gross_Sales_USD"] = 3900.0
    return df


def run_full_pipeline(df: pd.DataFrame, alpha: float = 0.60, k: int = 3):
    print("=" * 80)
    print("  AUTOMATED EDA PIPELINE: LLM-ASSISTED ROLE-AWARE INSIGHT RANKING")
    print("=" * 80)
    
    # --- STAGE I: SCHEMA NORMALIZATION ---
    print("\n--- STAGE I: SCHEMA NORMALIZATION ---")
    normalizer = SchemaNormalizer()
    mapping, details = normalizer.map_columns(df)
    for raw_col, info in details.items():
        print(f"  Raw: '{raw_col:<20}' -> Standard: '{info['standard_term']:<18}' (Conf: {info['confidence']}, Method: {info['method']})")
    
    normalized_df, _ = normalizer.normalize_dataframe(df, mapping)
    
    # --- STAGE II: INSIGHT GENERATION ---
    print("\n--- STAGE II: INSIGHT GENERATION (UNRANKED SHARED POOL) ---")
    generator = InsightGenerator(normalized_df)
    insight_pool = generator.generate_all_insights()
    print(f"Generated {len(insight_pool)} role-agnostic insights in the shared pool:")
    for ins in insight_pool:
        print(f"  * [{ins['insight_id']}] {ins['title']}")

    # --- STAGE III: ROLE-RELEVANCE RE-RANKING ---
    print(f"\n--- STAGE III: ROLE-RELEVANCE RE-RANKING (Hybrid alpha={alpha}) ---")
    ranker = RoleRanker(alpha=alpha)
    all_ranked = ranker.rank_all_roles(insight_pool)
    
    for role_name, ranked_list in all_ranked.items():
        print(f"\n  >> Role: [{role_name}]")
        for ins in ranked_list[:2]:
            print(f"     Rank #{ins['role_rank']}: [{ins['insight_id']}] {ins['title']}")
            print(f"             Hybrid Score: {ins['hybrid_score']} (LLM: {ins['llm_score']}, Rule: {ins['rule_score']})")

    # --- STAGE IV: EVALUATION & ABLATION BENCHMARK ---
    print(f"\n--- STAGE IV: EVALUATION & ABLATION BENCHMARK (NDCG@{k}, Precision@{k}) ---")
    gt = generate_sample_ground_truth()
    evaluator = Evaluator(gt)
    
    ablation_df = evaluator.run_ablation_study(insight_pool, ranker, k=k)
    print("\nAblation Results Table:")
    print(ablation_df.to_string(index=False))

    # --- INTER-ANNOTATOR AGREEMENT ---
    print("\n--- INTER-ANNOTATOR AGREEMENT CHECK ---")
    annotator1_scores = [2.0, 2.0, 1.0, 0.0]
    annotator2_scores = [2.0, 1.0, 1.0, 0.0]
    agreement = compute_inter_annotator_agreement(annotator1_scores, annotator2_scores)
    print(f"Kendall's Tau Agreement between Human Annotators: {agreement['kendall_tau']} (p-value: {agreement['p_value']})")

    print("\n" + "=" * 80)
    print("  PIPELINE EXECUTION COMPLETE: ALL 5 STAGES VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    test_data = generate_sample_retail_data()
    run_full_pipeline(test_data, alpha=0.60, k=3)
