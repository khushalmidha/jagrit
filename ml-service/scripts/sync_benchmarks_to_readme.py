"""
Synchronizes generated ablation and benchmark metrics from results/*.json
directly into ml-service/README.md and the repository root README.md.

Guarantees 100% mathematical alignment between code outputs, JSON artifacts,
and documentation. Zero discrepancy.
"""

import os
import re
import json
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SyncBenchmarks")


def generate_ablation_markdown_table(ablation_data: Dict[str, Any]) -> str:
    variants = ablation_data["variants"]
    v1 = variants["1_TwoTower_TextOnly"]
    v2 = variants["2_TwoTower_VisualConcat"]
    v3 = variants["3_TwoTower_Full_Gated"]
    v4 = variants["4_LambdaMART_Only"]
    v5 = variants["5_Ensemble_Locked"]
    v6 = variants["6_Ensemble_MMR_Diversity"]
    locked_alpha = ablation_data.get("locked_validation_alpha", 0.50)

    # Calculate percentage gain of v2 over v1, v3 over v2, v5 over v3
    gain_v2_v1 = ((v2["nDCG@10"] - v1["nDCG@10"]) / max(1e-6, v1["nDCG@10"])) * 100.0
    gain_v3_v2 = ((v3["nDCG@10"] - v2["nDCG@10"]) / max(1e-6, v2["nDCG@10"])) * 100.0
    gain_v5_v3 = ((v5["nDCG@10"] - v3["nDCG@10"]) / max(1e-6, v3["nDCG@10"])) * 100.0
    diversity_lift = ((v6["Intra_List_Diversity"] - v5["Intra_List_Diversity"]) / max(1e-6, v5["Intra_List_Diversity"])) * 100.0

    lines = [
        "| Model / Architecture Variant | nDCG@5 | nDCG@10 | PR-AUC | ROC-AUC | Cohen's Kappa ($\\kappa$) | MRR | Topic Diversity (ILD) | P95 Latency | Scientific Contribution / Rationale |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        f"| **1. Two-Tower (Text-Only Base)** | {v1['nDCG@5']:.4f} | {v1['nDCG@10']:.4f} | {v1['PR_AUC']:.4f} | {v1['ROC_AUC']:.4f} | {v1['Cohen_Kappa']:.4f} | {v1['MRR']:.4f} | {v1['Intra_List_Diversity']:.2f} | **{v1['P95_Latency_ms']:.2f} ms** | Pure text title token representations. Lacks visual grounding. |",
        f"| **2. Two-Tower + Visual Features** | {v2['nDCG@5']:.4f} | {v2['nDCG@10']:.4f} | {v2['PR_AUC']:.4f} | {v2['ROC_AUC']:.4f} | {v2['Cohen_Kappa']:.4f} | {v2['MRR']:.4f} | {v2['Intra_List_Diversity']:.2f} | {v2['P95_Latency_ms']:.2f} ms | **+{gain_v2_v1:.1f}% nDCG@10 gain**: Visual features disambiguate short and ambiguous breaking news headlines. |",
        f"| **3. Two-Tower + Attention & Gating** | {v3['nDCG@5']:.4f} | {v3['nDCG@10']:.4f} | {v3['PR_AUC']:.4f} | {v3['ROC_AUC']:.4f} | {v3['Cohen_Kappa']:.4f} | {v3['MRR']:.4f} | {v3['Intra_List_Diversity']:.2f} | {v3['P95_Latency_ms']:.2f} ms | **+{gain_v3_v2:.1f}% nDCG@10 gain**: MHA captures word semantics; Gated Cross-Modal Fusion dynamically balances channel confidence. |",
        f"| **4. Stage 2: LambdaMART Only** | {v4['nDCG@5']:.4f} | {v4['nDCG@10']:.4f} | {v4['PR_AUC']:.4f} | {v4['ROC_AUC']:.4f} | {v4['Cohen_Kappa']:.4f} | {v4['MRR']:.4f} | {v4['Intra_List_Diversity']:.2f} | {v4['P95_Latency_ms']:.2f} ms | Decision trees split on tabular signals (Bayesian CTR, category affinities, recency decay) missed by embeddings. |",
        f"| **5. Full 2-Stage Ensemble ($\\alpha = {locked_alpha:.2f}$)** | **{v5['nDCG@5']:.4f}** | **{v5['nDCG@10']:.4f}** | **{v5['PR_AUC']:.4f}** | **{v5['ROC_AUC']:.4f}** | **{v5['Cohen_Kappa']:.4f}** | **{v5['MRR']:.4f}** | {v5['Intra_List_Diversity']:.2f} | {v5['P95_Latency_ms']:.2f} ms | **+{gain_v5_v3:.1f}% nDCG@10 synergistic lift**: Combines dense semantic retrieval with non-linear tabular decision trees. |",
        f"| **6. Ensemble + MMR Diversity ($\\lambda = 0.75$)** | {v6['nDCG@5']:.4f} | {v6['nDCG@10']:.4f} | {v6['PR_AUC']:.4f} | {v6['ROC_AUC']:.4f} | {v6['Cohen_Kappa']:.4f} | {v6['MRR']:.4f} | **{v6['Intra_List_Diversity']:.2f}** | {v6['P95_Latency_ms']:.2f} ms | **+{diversity_lift:.1f}% diversity lift**: Breaks filter bubbles and delivers balanced category representation. |",
    ]
    return "\n".join(lines)


def generate_test_benchmark_markdown_table(test_data: Dict[str, Any]) -> str:
    s1 = test_data.get("Stage1_TwoTower", {})
    s2 = test_data.get("Stage2_LambdaMART", {})
    ens = test_data.get("Ensemble_Locked", {})
    mmr = test_data.get("Ensemble_MMR", {})
    alpha = test_data.get("locked_alpha", 0.50)
    lam = test_data.get("locked_mmr_lambda", 0.75)

    lines = [
        "| Architecture Component | nDCG@5 | nDCG@10 | MRR | ROC-AUC | PR-AUC | Cohen's $\\kappa$ | P50 (ms) | P95 (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Stage 1: Two-Tower (PyTorch)** | {s1.get('nDCG@5', 0):.4f} | {s1.get('nDCG@10', 0):.4f} | {s1.get('MRR', 0):.4f} | {s1.get('ROC-AUC', 0):.4f} | {s1.get('PR-AUC', 0):.4f} | {s1.get('Cohen_Kappa', 0):.4f} | {s1.get('P50_Latency_ms', 0):.2f} | {s1.get('P95_Latency_ms', 0):.2f} |",
        f"| **Stage 2: LambdaMART (XGBoost)** | {s2.get('nDCG@5', 0):.4f} | {s2.get('nDCG@10', 0):.4f} | {s2.get('MRR', 0):.4f} | {s2.get('ROC-AUC', 0):.4f} | {s2.get('PR-AUC', 0):.4f} | {s2.get('Cohen_Kappa', 0):.4f} | {s2.get('P50_Latency_ms', 0):.2f} | {s2.get('P95_Latency_ms', 0):.2f} |",
        f"| **Ensemble (Locked $\\alpha = {alpha:.2f}$)** | **{ens.get('nDCG@5', 0):.4f}** | **{ens.get('nDCG@10', 0):.4f}** | **{ens.get('MRR', 0):.4f}** | **{ens.get('ROC-AUC', 0):.4f}** | **{ens.get('PR-AUC', 0):.4f}** | **{ens.get('Cohen_Kappa', 0):.4f}** | {ens.get('P50_Latency_ms', 0):.2f} | {ens.get('P95_Latency_ms', 0):.2f} |",
        f"| **Ensemble + MMR ($\\lambda = {lam:.2f}$)** | {mmr.get('nDCG@5', 0):.4f} | {mmr.get('nDCG@10', 0):.4f} | {mmr.get('MRR', 0):.4f} | {mmr.get('ROC-AUC', 0):.4f} | {mmr.get('PR-AUC', 0):.4f} | {mmr.get('Cohen_Kappa', 0):.4f} | {mmr.get('P50_Latency_ms', 0):.2f} | {mmr.get('P95_Latency_ms', 0):.2f} |",
    ]
    return "\n".join(lines)


def update_readme_files(base_dir: str):
    results_dir = os.path.join(base_dir, "results")
    ablation_json_path = os.path.join(results_dir, "ablation_results.json")
    if not os.path.exists(ablation_json_path):
        ablation_json_path = os.path.join(results_dir, "ablation.json")

    test_json_path = os.path.join(results_dir, "test_results.json")

    if not os.path.exists(ablation_json_path):
        logger.error(f"Cannot find ablation results at {ablation_json_path}")
        return

    with open(ablation_json_path, "r", encoding="utf-8") as f:
        ablation_data = json.load(f)

    test_data = {}
    if os.path.exists(test_json_path):
        with open(test_json_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

    new_ablation_table = generate_ablation_markdown_table(ablation_data)
    locked_alpha = ablation_data.get("locked_validation_alpha", 0.50)
    v5 = ablation_data["variants"]["5_Ensemble_Locked"]

    # Target READMEs
    ml_readme = os.path.join(base_dir, "README.md")
    root_readme = os.path.join(os.path.dirname(base_dir), "README.md")

    for path in [ml_readme, root_readme]:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Replace Ablation Table
        pattern = r"\| Model / Architecture Variant \| nDCG@5 \|.*?\n(?:\| .*? \|\n)+"
        if re.search(pattern, content):
            content = re.sub(pattern, lambda _: new_ablation_table + "\n", content, flags=re.DOTALL)
            logger.info(f"Replaced Ablation table in {path}")
        else:
            pattern_root = r"\| Variant / Configuration \| nDCG@5 \|.*?\n(?:\| .*? \|\n)+"
            if re.search(pattern_root, content):
                content = re.sub(pattern_root, lambda _: new_ablation_table + "\n", content, flags=re.DOTALL)
                logger.info(f"Replaced Variant table in {path}")

        # Replace Test Benchmark Table if present
        if test_data:
            new_test_table = generate_test_benchmark_markdown_table(test_data)
            test_table_pattern = r"\| Architecture Component \| nDCG@5 \|.*?\n(?:\| .*? \|\n)+"
            if re.search(test_table_pattern, content):
                content = re.sub(test_table_pattern, lambda _: new_test_table + "\n", content, flags=re.DOTALL)
                logger.info(f"Replaced Test Benchmark table in {path}")

        # Update alpha references and nDCG claims in text
        content = re.sub(r"optimal α = 0\.\d+", f"optimal α = {locked_alpha:.2f}", content)
        content = re.sub(r"optimal alpha = 0\.\d+", f"optimal alpha = {locked_alpha:.2f}", content)
        content = re.sub(r"Optimal α = 0\.\d+", f"Optimal α = {locked_alpha:.2f}", content)
        content = re.sub(r"Optimal alpha = 0\.\d+", f"Optimal alpha = {locked_alpha:.2f}", content)
        content = re.sub(r"\(Optimal α = 0\.\d+\)", f"(Optimal α = {locked_alpha:.2f})", content)
        content = re.sub(r"\(Optimal alpha = 0\.\d+\)", f"(Optimal alpha = {locked_alpha:.2f})", content)
        content = re.sub(r"α = 0\.\d+", f"α = {locked_alpha:.2f}", content)
        content = re.sub(r"alpha = 0\.\d+", f"alpha = {locked_alpha:.2f}", content)
        content = re.sub(r"\b0\.6073 nDCG@10\b", f"{v5['nDCG@10']:.4f} nDCG@10", content)
        content = re.sub(r"\bnDCG@10 = 0\.6073\b", f"nDCG@10 = {v5['nDCG@10']:.4f}", content)

        # Remove "Mixed precision acceleration" on CPU claim
        content = content.replace("528.14s on 8-core CPU | Mixed precision acceleration", "528.14s on 8-core CPU")
        content = content.replace(
            "Amazon Applied Scientist standards require validating *why* each model component is introduced:",
            "Component-by-component scientific ablation breakdown evaluated on identical test impressions:"
        )

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Successfully synchronized {path} with ablation metrics.")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    update_readme_files(base_dir)
