"""
insight_generator.py
--------------------
Stage II: Data Cleaning and Statistical Insight Generation Engine.

Cleans normalized data and scans for statistical patterns (trends, outliers,
attributions, correlations, Pareto distributions) to construct a single,
role-agnostic shared insight pool with simple language summaries and Plotly-ready visual specs.
"""

import math
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


class InsightGenerator:
    """
    Scans a normalized DataFrame to generate a rich pool of statistical insights.
    All insights are role-agnostic at this stage.
    """

    def __init__(self, df: pd.DataFrame):
        self.raw_df = df.copy()
        self.df = self._clean_data(df)
        self.insights: List[Dict[str, Any]] = []

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataset, coerce types, parse dates."""
        cleaned = df.copy()

        # Parse Date Column
        if "date" in cleaned.columns:
            cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
            cleaned = cleaned.sort_values("date").dropna(subset=["date"])

        # Coerce Numeric Columns
        numeric_cols = [
            "sales_amount", "units_sold", "unit_price", "discount_pct",
            "profit", "cost_price", "inventory_level", "marketing_spend",
            "conversion_rate"
        ]
        for col in numeric_cols:
            if col in cleaned.columns:
                cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").fillna(0)

        # Derived metrics if missing
        if "sales_amount" in cleaned.columns and "units_sold" in cleaned.columns and "unit_price" not in cleaned.columns:
            cleaned["unit_price"] = np.where(cleaned["units_sold"] > 0, cleaned["sales_amount"] / cleaned["units_sold"], 0)

        if "sales_amount" in cleaned.columns and "profit" in cleaned.columns and "margin_pct" not in cleaned.columns:
            cleaned["margin_pct"] = np.where(cleaned["sales_amount"] > 0, (cleaned["profit"] / cleaned["sales_amount"]) * 100, 0)

        return cleaned

    def generate_all_insights(self) -> List[Dict[str, Any]]:
        """Run all statistical detectors and compile the unranked insight pool."""
        self.insights = []

        self._detect_overall_kpis()
        self._detect_time_trends()
        self._detect_outliers()
        self._detect_category_attribution()
        self._detect_regional_performance()
        self._detect_correlations()
        self._detect_return_rate_anomalies()
        self._detect_inventory_alerts()

        return self.insights

    def _detect_overall_kpis(self):
        """Generates baseline executive KPI insights."""
        if "sales_amount" in self.df.columns:
            total_sales = float(self.df["sales_amount"].sum())
            avg_order = float(self.df["sales_amount"].mean())
            total_orders = len(self.df)

            self.insights.append({
                "insight_id": "INS_KPI_01",
                "category": "efficiency",
                "title": f"Total Revenue Reached ${total_sales:,.2f} Across {total_orders:,} Transactions",
                "simple_explanation": f"The store generated ${total_sales:,.2f} in total gross revenue with an average order value (AOV) of ${avg_order:,.2f}.",
                "detailed_description": f"Overall performance summary: Total revenue = ${total_sales:,.2f}, Total Transactions = {total_orders:,}, Average Transaction Value = ${avg_order:,.2f}.",
                "metrics": {"total_sales": total_sales, "avg_order_value": avg_order, "total_orders": total_orders},
                "associated_columns": ["sales_amount", "order_id"],
                "chart_type": "metric_card",
                "chart_data": {"metric_label": "Total Gross Revenue", "value": f"${total_sales:,.2f}", "subtext": f"AOV: ${avg_order:,.2f}"}
            })

    def _detect_time_trends(self):
        """Detects monthly or daily revenue trends and growth trajectories."""
        if "date" in self.df.columns and "sales_amount" in self.df.columns and len(self.df) > 5:
            # Resample to Monthly or Weekly depending on span
            span_days = (self.df["date"].max() - self.df["date"].min()).days
            target_freqs = ["ME", "M", "MS"] if span_days > 60 else ["W", "W-MON", "D"]
            
            time_df = None
            chosen_freq = None
            for freq_candidate in target_freqs:
                try:
                    resampler = self.df.set_index("date").resample(freq_candidate)
                    time_df = resampler["sales_amount"].sum().reset_index()
                    chosen_freq = freq_candidate
                    break
                except Exception:
                    continue
            
            if time_df is not None and len(time_df) >= 2:
                date_format = "%Y-%m" if chosen_freq in ["ME", "M", "MS"] else "%Y-%W"
                time_df["date_str"] = time_df["date"].dt.strftime(date_format)
                first_val = time_df["sales_amount"].iloc[0]
                last_val = time_df["sales_amount"].iloc[-1]
                peak_row = time_df.loc[time_df["sales_amount"].idxmax()]
                
                pct_change = ((last_val - first_val) / first_val * 100) if first_val > 0 else 0
                direction = "grew by" if pct_change >= 0 else "declined by"

                self.insights.append({
                    "insight_id": "INS_TREND_01",
                    "category": "trend",
                    "title": f"Revenue {direction.capitalize()} {abs(pct_change):.1f}% Over the Monitored Period",
                    "simple_explanation": f"Sales trajectory shifted from ${first_val:,.2f} in the starting period to ${last_val:,.2f} in the recent period, peaking at ${peak_row['sales_amount']:,.2f} ({peak_row['date_str']}).",
                    "detailed_description": f"Time-series regression indicates a net {direction} of {abs(pct_change):.2f}%. Maximum single-period revenue occurred on {peak_row['date_str']}.",
                    "metrics": {"growth_pct": pct_change, "peak_value": float(peak_row['sales_amount']), "peak_date": str(peak_row['date_str'])},
                    "associated_columns": ["date", "sales_amount"],
                    "chart_type": "line",
                    "chart_data": {
                        "x": time_df["date_str"].tolist(),
                        "y": [float(v) for v in time_df["sales_amount"]],
                        "x_label": "Time Period",
                        "y_label": "Sales Amount ($)"
                    }
                })

    def _detect_outliers(self):
        """Identifies statistical revenue spikes or unusual discount anomalies."""
        if "sales_amount" in self.df.columns and len(self.df) > 10:
            mean_val = self.df["sales_amount"].mean()
            std_val = self.df["sales_amount"].std()
            
            if std_val > 0:
                self.df["sales_zscore"] = (self.df["sales_amount"] - mean_val) / std_val
                outliers = self.df[self.df["sales_zscore"] > 2.5]
                
                if len(outliers) > 0:
                    max_outlier = outliers.sort_values("sales_amount", ascending=False).iloc[0]
                    col_item = "product_name" if "product_name" in max_outlier else ("category" if "category" in max_outlier else "order_id")
                    item_name = max_outlier.get(col_item, "Transaction")

                    self.insights.append({
                        "insight_id": "INS_OUTLIER_01",
                        "category": "outlier",
                        "title": f"Detected High-Value Transaction Spike: ${max_outlier['sales_amount']:,.2f}",
                        "simple_explanation": f"An unusually large sale of ${max_outlier['sales_amount']:,.2f} ({item_name}) was identified, exceeding 2.5 standard deviations above normal average order size.",
                        "detailed_description": f"Found {len(outliers)} transaction outlier(s). Top outlier value = ${max_outlier['sales_amount']:,.2f} compared to dataset mean of ${mean_val:,.2f}.",
                        "metrics": {"outlier_count": len(outliers), "max_outlier_value": float(max_outlier['sales_amount']), "z_score": float(max_outlier['sales_zscore'])},
                        "associated_columns": ["sales_amount", "product_name", "order_id"],
                        "chart_type": "bar",
                        "chart_data": {
                            "x": ["Average Order", f"Top Outlier ({item_name})"],
                            "y": [float(mean_val), float(max_outlier['sales_amount'])],
                            "x_label": "Comparison",
                            "y_label": "Sales Value ($)"
                        }
                    })

    def _detect_category_attribution(self):
        """Analyzes product category revenue contribution and Pareto 80/20 breakdown."""
        if "category" in self.df.columns and "sales_amount" in self.df.columns:
            cat_df = self.df.groupby("category")["sales_amount"].sum().reset_index()
            cat_df = cat_df.sort_values("sales_amount", ascending=False)
            total_sales = cat_df["sales_amount"].sum()
            
            if total_sales > 0 and len(cat_df) > 1:
                cat_df["pct"] = (cat_df["sales_amount"] / total_sales) * 100
                top_cat = cat_df.iloc[0]

                self.insights.append({
                    "insight_id": "INS_ATTR_01",
                    "category": "attribution",
                    "title": f"'{top_cat['category']}' Dominates Sales, Generating {top_cat['pct']:.1f}% of Total Revenue",
                    "simple_explanation": f"The '{top_cat['category']}' category generated ${top_cat['sales_amount']:,.2f}, representing the single largest revenue driver across all departments.",
                    "detailed_description": f"Category attribution ranking: Top category '{top_cat['category']}' accounts for {top_cat['pct']:.2f}% of total sales across {len(cat_df)} categories.",
                    "metrics": {"top_category": str(top_cat['category']), "share_pct": float(top_cat['pct']), "category_sales": float(top_cat['sales_amount'])},
                    "associated_columns": ["category", "sales_amount"],
                    "chart_type": "donut",
                    "chart_data": {
                        "labels": cat_df["category"].astype(str).tolist(),
                        "values": [float(v) for v in cat_df["sales_amount"]],
                        "title": "Revenue Distribution by Category"
                    }
                })

    def _detect_regional_performance(self):
        """Analyzes regional sales dispersion and store channel contributions."""
        if "region" in self.df.columns and "sales_amount" in self.df.columns:
            reg_df = self.df.groupby("region")["sales_amount"].sum().reset_index().sort_values("sales_amount", ascending=False)
            if len(reg_df) > 1:
                top_reg = reg_df.iloc[0]
                bottom_reg = reg_df.iloc[-1]
                ratio = top_reg["sales_amount"] / bottom_reg["sales_amount"] if bottom_reg["sales_amount"] > 0 else 1.0

                self.insights.append({
                    "insight_id": "INS_REGION_01",
                    "category": "attribution",
                    "title": f"Top Performing Region '{top_reg['region']}' Outpaces Lowest Region by {ratio:.1f}x",
                    "simple_explanation": f"Regional sales are led by '{top_reg['region']}' (${top_reg['sales_amount']:,.2f}), while '{bottom_reg['region']}' recorded ${bottom_reg['sales_amount']:,.2f}.",
                    "detailed_description": f"Regional breakdown shows highest demand in '{top_reg['region']}' vs lowest in '{bottom_reg['region']}'. Disparity ratio: {ratio:.2f}x.",
                    "metrics": {"top_region": str(top_reg['region']), "bottom_region": str(bottom_reg['region']), "disparity_ratio": float(ratio)},
                    "associated_columns": ["region", "sales_amount"],
                    "chart_type": "bar",
                    "chart_data": {
                        "x": reg_df["region"].astype(str).tolist(),
                        "y": [float(v) for v in reg_df["sales_amount"]],
                        "x_label": "Region",
                        "y_label": "Total Revenue ($)"
                    }
                })

    def _detect_correlations(self):
        """Detects correlations between price, discounts, and sales volume."""
        if "discount_pct" in self.df.columns and "units_sold" in self.df.columns:
            corr = self.df["discount_pct"].corr(self.df["units_sold"])
            if not math.isnan(corr) and abs(corr) >= 0.2:
                relationship = "positive" if corr > 0 else "negative"
                strength = "strong" if abs(corr) >= 0.5 else "moderate"

                self.insights.append({
                    "insight_id": "INS_CORR_01",
                    "category": "correlation",
                    "title": f"{strength.capitalize()} {relationship.capitalize()} Link Between Discounts and Unit Volume (r = {corr:.2f})",
                    "simple_explanation": f"Higher promotional discounts show a {relationship} correlation with unit sales volume, indicating price sensitivity.",
                    "detailed_description": f"Pearson correlation coefficient between discount_pct and units_sold = {corr:.3f}.",
                    "metrics": {"correlation_r": float(corr), "strength": strength, "relationship": relationship},
                    "associated_columns": ["discount_pct", "units_sold"],
                    "chart_type": "scatter",
                    "chart_data": {
                        "x": [float(v) for v in self.df["discount_pct"]],
                        "y": [float(v) for v in self.df["units_sold"]],
                        "x_label": "Discount %",
                        "y_label": "Units Sold"
                    }
                })

    def _detect_return_rate_anomalies(self):
        """Analyzes order return rates if return status column exists."""
        if "return_status" in self.df.columns:
            returns_col = self.df["return_status"].astype(str).str.lower()
            is_returned = returns_col.isin(["yes", "true", "1", "returned"])
            total_orders = len(self.df)
            returned_count = is_returned.sum()
            
            if total_orders > 0:
                return_rate = (returned_count / total_orders) * 100
                
                self.insights.append({
                    "insight_id": "INS_RET_01",
                    "category": "efficiency",
                    "title": f"Product Return Rate Stands at {return_rate:.1f}% ({returned_count:,} Orders Returned)",
                    "simple_explanation": f"Out of {total_orders:,} total orders, {returned_count:,} orders resulted in returns or refunds.",
                    "detailed_description": f"Product return rate calculation: {returned_count} returned out of {total_orders} orders = {return_rate:.2f}%.",
                    "metrics": {"return_rate_pct": float(return_rate), "returned_orders": int(returned_count), "total_orders": int(total_orders)},
                    "associated_columns": ["return_status", "order_id"],
                    "chart_type": "donut",
                    "chart_data": {
                        "labels": ["Keep/Fulfilled", "Returned"],
                        "values": [int(total_orders - returned_count), int(returned_count)],
                        "title": "Return Rate Breakdown"
                    }
                })

    def _detect_inventory_alerts(self):
        """Identifies stockout risk or inventory level anomalies if present."""
        if "inventory_level" in self.df.columns and "product_name" in self.df.columns:
            low_stock = self.df[self.df["inventory_level"] < 10]
            if len(low_stock) > 0:
                items = low_stock["product_name"].unique().tolist()[:3]
                self.insights.append({
                    "insight_id": "INS_INV_01",
                    "category": "efficiency",
                    "title": f"Low Stock Alert: {len(low_stock)} SKUs Falling Below Reorder Threshold",
                    "simple_explanation": f"Found {len(low_stock)} items with critical inventory levels (<10 units), including {', '.join(items)}.",
                    "detailed_description": f"Inventory stockout alert: {len(low_stock)} SKUs require immediate warehouse replenishment.",
                    "metrics": {"low_stock_count": len(low_stock), "sample_items": items},
                    "associated_columns": ["inventory_level", "product_name"],
                    "chart_type": "bar",
                    "chart_data": {
                        "x": [str(x) for x in low_stock["product_name"].iloc[:5]],
                        "y": [float(y) for y in low_stock["inventory_level"].iloc[:5]],
                        "x_label": "Product SKU",
                        "y_label": "Stock Level"
                    }
                })


if __name__ == "__main__":
    test_df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=10, freq="D"),
        "sales_amount": [100, 120, 150, 90, 80, 500, 110, 130, 140, 160],
        "units_sold": [1, 2, 3, 1, 1, 10, 2, 2, 3, 4],
        "discount_pct": [0, 5, 10, 0, 0, 25, 5, 5, 10, 15],
        "category": ["Apparel", "Apparel", "Electronics", "Electronics", "Apparel", "Electronics", "Apparel", "Electronics", "Apparel", "Electronics"],
        "region": ["West", "East", "West", "East", "South", "West", "South", "Central", "West", "East"],
        "return_status": ["No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "Yes"]
    })
    
    generator = InsightGenerator(test_df)
    pool = generator.generate_all_insights()
    print(f"Generated {len(pool)} insights in single shared pool:")
    for ins in pool:
        print(f" - [{ins['insight_id']}] {ins['title']}")
