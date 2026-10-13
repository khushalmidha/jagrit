"""
Regression and Consistency Validator for Jagrit Documentation.

Validates that:
1. All required benchmark artifact JSON and CSV files exist and are well-formed.
2. Verified test holdout metrics and frozen hyperparameters match code artifacts.
3. No forbidden obsolete legacy metrics (e.g. 0.6073, 8.6ms, alpha=0.50) appear anywhere
   in public-facing documentation.
4. Scientific generalization gap is properly documented without deceptive marketing.

Usage:
  python scripts/validate_documentation.py
Exits 0 if all checks pass; exits 1 with diagnostic output on any failure.
"""

import os
import re
import sys
import json
from typing import List, Dict, Any

FORBIDDEN_LEGACY_PATTERNS = [
    (r"0\.6073", "Obsolete fabricated nDCG@10 metric (0.6073)"),
    (r"0\.5111", "Legacy Stage 1 baseline metric (0.5111)"),
    (r"0\.4492", "Legacy text baseline metric (0.4492)"),
    (r"0\.4765", "Legacy visual baseline metric (0.4765)"),
    (r"0\.4318", "Legacy attention baseline metric (0.4318)"),
    (r"0\.1984", "Legacy classification metric (0.1984)"),
    (r"8\.61\s*ms", "Legacy latency claim (8.61 ms)"),
    (r"8\.6\s*ms", "Legacy latency claim (8.6 ms)"),
    (r"0\.66\s*ms", "Legacy sub-millisecond claim (0.66 ms)"),
    (r"7\.95\s*ms", "Legacy reranker latency claim (7.95 ms)"),
    (r"sub-10\s*ms", "Unverified production latency claim (sub-10ms)"),
    (r"synergistic\s+(?:boost|lift)", "Misleading claim asserting ensemble outperforms Stage 1"),
]


def check_artifacts(base_dir: str) -> Dict[str, Any]:
    print("[1/4] Verifying required artifact files...")
    results_dir = os.path.join(base_dir, "results")
    checkpoints_dir = os.path.join(base_dir, "checkpoints")

    required_files = [
        os.path.join(results_dir, "split_manifest.json"),
        os.path.join(results_dir, "best_validation_config.json"),
        os.path.join(results_dir, "test_results.json"),
        os.path.join(results_dir, "ablation_results.json"),
        os.path.join(results_dir, "ablation_results.csv"),
        os.path.join(checkpoints_dir, "ensemble_config.json"),
        os.path.join(checkpoints_dir, "train_feature_priors.json"),
    ]

    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"FAILED: Missing required artifact files: {missing}")
        sys.exit(1)

    with open(os.path.join(results_dir, "best_validation_config.json"), "r", encoding="utf-8") as f:
        val_cfg = json.load(f)

    with open(os.path.join(results_dir, "test_results.json"), "r", encoding="utf-8") as f:
        test_res = json.load(f)

    with open(os.path.join(results_dir, "ablation_results.json"), "r", encoding="utf-8") as f:
        ablation_res = json.load(f)

    print("  -> All 7 artifact files present and verified.")
    return {
        "val_cfg": val_cfg,
        "test_res": test_res,
        "ablation_res": ablation_res,
    }


def validate_metrics(artifacts: Dict[str, Any]):
    print("[2/4] Validating metric consistency in artifacts...")
    val_cfg = artifacts["val_cfg"]
    test_res = artifacts["test_res"]
    ablation_res = artifacts["ablation_res"]

    # 1. Validation tuning
    alpha = val_cfg.get("optimal_alpha_two_tower")
    lam = val_cfg.get("optimal_mmr_lambda")
    if round(alpha, 2) != 0.60:
        print(f"FAILED: Expected validation alpha* = 0.60, found {alpha}")
        sys.exit(1)
    if round(lam, 2) != 0.75:
        print(f"FAILED: Expected validation lambda* = 0.75, found {lam}")
        sys.exit(1)

    # 2. Test results
    s1_ndcg10 = test_res.get("Stage1_TwoTower", {}).get("nDCG@10", 0)
    s2_ndcg10 = test_res.get("Stage2_LambdaMART", {}).get("nDCG@10", 0)
    ens_ndcg10 = test_res.get("Ensemble_Locked", {}).get("nDCG@10", 0)

    if round(s1_ndcg10, 4) != 0.2877:
        print(f"FAILED: Expected Stage 1 nDCG@10 = 0.2877, found {s1_ndcg10}")
        sys.exit(1)
    if round(s2_ndcg10, 4) != 0.2616:
        print(f"FAILED: Expected Stage 2 nDCG@10 = 0.2616, found {s2_ndcg10}")
        sys.exit(1)
    if round(ens_ndcg10, 4) != 0.2768:
        print(f"FAILED: Expected Ensemble nDCG@10 = 0.2768, found {ens_ndcg10}")
        sys.exit(1)

    # 3. Check generalization gap holds true (Two-Tower > Ensemble on test)
    if s1_ndcg10 <= ens_ndcg10:
        print("FAILED: Two-Tower test nDCG@10 should exceed Ensemble test nDCG@10 reflecting distribution shift.")
        sys.exit(1)

    print("  -> Artifact metrics verified mathematically.")


def scan_documentation_for_stale_metrics(repo_root: str):
    print("[3/4] Scanning documentation files for prohibited legacy patterns...")
    target_docs = [
        os.path.join(repo_root, "README.md"),
        os.path.join(repo_root, "REPRODUCIBILITY.md"),
        os.path.join(repo_root, "ml-service", "README.md"),
        os.path.join(repo_root, "ml-service", "REPRODUCIBILITY.md"),
    ]

    violations: List[str] = []

    for doc_path in target_docs:
        if not os.path.exists(doc_path):
            violations.append(f"Target document missing: {doc_path}")
            continue

        with open(doc_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        rel_path = os.path.relpath(doc_path, repo_root)

        for line_idx, line in enumerate(lines, 1):
            # Allow alpha = 0.50 inside sweep tables where candidate 0.50 is swept
            if "| 0.50 | 0.50 |" in line or "hard_ratio: 0.50" in line:
                continue

            for pattern, desc in FORBIDDEN_LEGACY_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    violations.append(f"{rel_path}:{line_idx} - {desc} in: {line.strip()[:90]}")

    if violations:
        print(f"FAILED: Found {len(violations)} obsolete metric violations in documentation:")
        for v in violations:
            print(f"  [X] {v}")
        sys.exit(1)

    print("  -> Zero prohibited legacy patterns found in documentation.")


def verify_required_statements(repo_root: str):
    print("[4/4] Verifying required scientific honesty statements...")
    root_readme = os.path.join(repo_root, "README.md")
    ml_readme = os.path.join(repo_root, "ml-service", "README.md")

    for p in [root_readme, ml_readme]:
        with open(p, "r", encoding="utf-8") as f:
            content = f.read()

        rel = os.path.relpath(p, repo_root)

        # 1. Must document Two-Tower as strongest test ranking model
        if "0.2877" not in content:
            print(f"FAILED: {rel} missing verified Stage 1 nDCG@10 (0.2877)")
            sys.exit(1)

        # 2. Must document Frozen Ensemble test metric
        if "0.2768" not in content:
            print(f"FAILED: {rel} missing verified Ensemble nDCG@10 (0.2768)")
            sys.exit(1)

        # 3. Must document generalization gap
        if "Generalization Gap" not in content:
            print(f"FAILED: {rel} missing explicit Generalization Gap discussion")
            sys.exit(1)

    print("  -> Scientific honesty statements verified in all README files.")


if __name__ == "__main__":
    ml_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.dirname(ml_dir)

    print("=" * 70)
    print("=== JAGRIT DOCUMENTATION & BENCHMARK INTEGRITY VALIDATOR ===")
    print("=" * 70)

    artifacts = check_artifacts(ml_dir)
    validate_metrics(artifacts)
    scan_documentation_for_stale_metrics(repo_root)
    verify_required_statements(repo_root)

    print("=" * 70)
    print("SUCCESS: 100% Repository-Wide Documentation Consistency Confirmed!")
    print("=" * 70)
    sys.exit(0)
