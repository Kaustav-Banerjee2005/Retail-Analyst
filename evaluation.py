"""
evaluation.py
-------------
Stage IV & V: Quantitative Evaluation & Ablation Engine.

Computes IR metrics (NDCG@K, Precision@K), Kendall's tau inter-annotator agreement,
and executes ablation studies (Hybrid vs. LLM-only vs. Rule-only) and generalization tests.
"""

import math
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def compute_dcg_at_k(relevance_scores: List[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at rank K."""
    relevance_scores = relevance_scores[:k]
    if not relevance_scores:
        return 0.0
    dcg = sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(relevance_scores))
    return dcg


def compute_ndcg_at_k(predicted_relevance: List[float], ideal_relevance: List[float], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank K."""
    dcg = compute_dcg_at_k(predicted_relevance, k)
    idcg = compute_dcg_at_k(sorted(ideal_relevance, reverse=True), k)
    if idcg == 0.0:
        return 0.0
    return round(dcg / idcg, 4)


def compute_precision_at_k(predicted_ids: List[str], ground_truth_relevant_ids: List[str], k: int) -> float:
    """Compute Precision@K (proportion of predicted top-K that are ground-truth relevant)."""
    top_k_pred = set(predicted_ids[:k])
    relevant_set = set(ground_truth_relevant_ids)
    if not top_k_pred:
        return 0.0
    overlap = len(top_k_pred.intersection(relevant_set))
    return round(overlap / min(k, len(top_k_pred)), 4)


def compute_inter_annotator_agreement(ratings_annotator1: List[float], ratings_annotator2: List[float]) -> Dict[str, float]:
    """Compute Kendall's tau rank correlation between two human annotators."""
    tau, p_value = kendalltau(ratings_annotator1, ratings_annotator2)
    return {
        "kendall_tau": round(float(tau), 4) if not math.isnan(tau) else 0.0,
        "p_value": round(float(p_value), 4) if not math.isnan(p_value) else 1.0
    }


class Evaluator:
    """
    Evaluation suite for testing role-aware re-ranking quality, ablation performance,
    and cross-dataset generalization.
    """

    def __init__(self, ground_truth: Dict[str, Dict[str, float]]):
        """
        ground_truth structure:
          {
            "Business Owner": {"INS_TREND_01": 2.0, "INS_KPI_01": 2.0, "INS_ATTR_01": 1.0, ...},
            "Store Manager": {"INS_REGION_01": 2.0, "INS_OUTLIER_01": 2.0, ...}, ...
          }
        0 = Irrelevant, 1 = Relevant, 2 = Highly Relevant
        """
        self.ground_truth = ground_truth

    def evaluate_role_ranking(
        self,
        ranked_insights: List[Dict[str, Any]],
        role_name: str,
        k: int = 3
    ) -> Dict[str, float]:
        """Evaluates NDCG@K and Precision@K for a single role's ranked output."""
        gt_role = self.ground_truth.get(role_name, {})
        if not gt_role:
            return {"ndcg_at_k": 0.0, "precision_at_k": 0.0}

        pred_ids = [ins["insight_id"] for ins in ranked_insights]
        pred_rel_scores = [gt_role.get(i_id, 0.0) for i_id in pred_ids]
        ideal_rel_scores = list(gt_role.values())

        gt_relevant_ids = [i_id for i_id, score in gt_role.items() if score >= 1.0]

        ndcg = compute_ndcg_at_k(pred_rel_scores, ideal_rel_scores, k=k)
        precision = compute_precision_at_k(pred_ids, gt_relevant_ids, k=k)

        return {
            "ndcg_at_k": ndcg,
            "precision_at_k": precision,
            "top_k_evaluated": k
        }

    def run_ablation_study(
        self,
        insight_pool: List[Dict[str, Any]],
        ranker_instance: Any,
        k: int = 3
    ) -> pd.DataFrame:
        """
        Runs Ablation Benchmark comparing:
          - Hybrid System (alpha = 0.60)
          - LLM-Only System (alpha = 1.00)
          - Rules-Only System (alpha = 0.00)
        """
        configs = {
            "Hybrid System (alpha=0.60)": 0.60,
            "LLM-Only System (alpha=1.00)": 1.00,
            "Rules-Only System (alpha=0.00)": 0.00
        }

        records = []
        for config_name, alpha in configs.items():
            for role_name in self.ground_truth.keys():
                ranked = ranker_instance.rank_insights_for_role(insight_pool, role_name, override_alpha=alpha)
                res = self.evaluate_role_ranking(ranked, role_name, k=k)
                records.append({
                    "Configuration": config_name,
                    "Role": role_name,
                    "NDCG@K": res["ndcg_at_k"],
                    "Precision@K": res["precision_at_k"]
                })

        return pd.DataFrame(records)


def generate_sample_ground_truth() -> Dict[str, Dict[str, float]]:
    """Synthetic ground truth matrix for benchmark testing."""
    return {
        "Business Owner": {
            "INS_KPI_01": 2.0,
            "INS_TREND_01": 2.0,
            "INS_ATTR_01": 1.0,
            "INS_REGION_01": 1.0,
            "INS_CORR_01": 0.0,
            "INS_RET_01": 0.0
        },
        "Store Manager": {
            "INS_REGION_01": 2.0,
            "INS_OUTLIER_01": 2.0,
            "INS_RET_01": 1.0,
            "INS_KPI_01": 1.0,
            "INS_TREND_01": 0.0,
            "INS_CORR_01": 0.0
        },
        "Marketing Head": {
            "INS_CORR_01": 2.0,
            "INS_ATTR_01": 2.0,
            "INS_TREND_01": 1.0,
            "INS_KPI_01": 1.0,
            "INS_REGION_01": 0.0,
            "INS_RET_01": 0.0
        },
        "Inventory Manager": {
            "INS_RET_01": 2.0,
            "INS_INV_01": 2.0,
            "INS_ATTR_01": 1.0,
            "INS_OUTLIER_01": 1.0,
            "INS_KPI_01": 0.0,
            "INS_TREND_01": 0.0
        }
    }


if __name__ == "__main__":
    from insight_generator import InsightGenerator
    from role_ranker import RoleRanker

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
    gt = generate_sample_ground_truth()

    evaluator = Evaluator(gt)
    ablation_df = evaluator.run_ablation_study(pool, ranker, k=3)

    print("--- ABLATION BENCHMARK RESULTS ---")
    print(ablation_df.to_string(index=False))

    # Test Kendall's Tau
    a1 = [2.0, 2.0, 1.0, 0.0]
    a2 = [2.0, 1.0, 1.0, 0.0]
    agreement = compute_inter_annotator_agreement(a1, a2)
    print("\nInter-Annotator Agreement (Kendall's Tau):", agreement)
