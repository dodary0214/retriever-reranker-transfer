"""Reversal rates, excess over C0, direction, strata, control check, verdict.
in: conditions.jsonl, pairs.tsv, rank_out.jsonl (or --rank FILE), ce_out.jsonl
out: results/table1_reversal.csv, table2_direction.csv, table3_strata.csv, table_supp.csv, summary.md, per_call.parquet"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
from config import EXP, SEED
import numpy as np, pandas as pd
from scipy.stats import binomtest

P1 = EXP / "controlled"
R = P1 / "results"
N_BOOT = 1000
CONDS = ["R", "C0", "C1", "C2", "C3", "C4", "C5"]
TESTED = ["C1", "C2", "C3", "C4", "C5"]


def load(rank_file):
    cond = pd.read_json(P1 / "conditions.jsonl", lines=True)
    pairs = pd.read_csv(P1 / "pairs.tsv", sep="\t", dtype={"docA": str, "docB": str})
    out = pd.read_json(rank_file, lines=True, dtype={"ranking": object})
    df = cond.merge(pairs, on=["qid", "pair_id"]).merge(out, on="call_id", how="inner")
    df["rankA"] = [r.index(a) + 1 for r, a in zip(df.ranking, df.docA)]
    df["rankB"] = [r.index(b) + 1 for r, b in zip(df.ranking, df.docB)]
    return df


def reversal_table(df):
    """One row per unit: (qid, pair_id, cond, rep[, perm]) with rev in {0, .5, 1}, plus the after-order."""
    base = df[df.cond == "B"].set_index(["qid", "pair_id"])
    base = base.assign(pref0=base.rankA < base.rankB, gap0=(base.rankA - base.rankB).abs())[["pref0", "gap0"]]
    d = df[df.cond.isin(["R", "C0", "C1", "C2", "C3", "C4"])].copy()
    d["a_above"] = (d.rankA < d.rankB).astype(float)
    c5 = (df[df.cond == "C5"].groupby(["qid", "pair_id", "rep", "gradeA", "gradeB"], as_index=False)
          .agg(rankA=("rankA", "mean"), rankB=("rankB", "mean")))
    c5["cond"] = "C5"
    c5["a_above"] = np.where(c5.rankA < c5.rankB, 1.0, np.where(c5.rankA > c5.rankB, 0.0, 0.5))
    u = pd.concat([d[["qid", "pair_id", "cond", "rep", "gradeA", "gradeB", "a_above"]],
                   c5[["qid", "pair_id", "cond", "rep", "gradeA", "gradeB", "a_above"]]], ignore_index=True)
    u = u.join(base, on=["qid", "pair_id"])
    u = u[u.pref0.notna()].copy()  # units whose base call is missing (partial runs) have no reference order
    u["pref0"] = u.pref0.astype(bool)
    u["rev"] = np.where(u.a_above == 0.5, 0.5, (u.a_above == 1.0) != u.pref0).astype(float)
    u["unequal"] = u.gradeA != u.gradeB
    u["gdiff"] = (u.gradeA - u.gradeB).abs()
    # after reversal the order is the opposite of pref0; it is "correct" if A (higher grade) is above
    u["after_correct"] = np.where(u.unequal, ~u.pref0.astype(bool), np.nan)
    u["pref0_correct"] = np.where(u.unequal, u.pref0.astype(bool), np.nan)
    return u


class Boot:
    """Query-level bootstrap over per-query sums (rate = mean over pairs of pair-level means)."""

    def __init__(self, queries):
        self.q = sorted(queries)
        rng = np.random.default_rng(SEED)
        self.idx = rng.integers(0, len(self.q), size=(N_BOOT, len(self.q)))

    def mats(self, u, value, by="cond"):
        pair = u.groupby(["qid", "pair_id", by])[value].mean().reset_index()
        s = pair.pivot_table(index="qid", columns=by, values=value, aggfunc="sum").reindex(self.q)
        n = pair.pivot_table(index="qid", columns=by, values=value, aggfunc="count").reindex(self.q)
        return s.fillna(0), n.fillna(0)

    def rates(self, s, n):
        S, N = s.values[self.idx].sum(1), n.values[self.idx].sum(1)  # (boot, cols)
        with np.errstate(invalid="ignore", divide="ignore"):
            return pd.DataFrame(S / N, columns=s.columns)


def ci(x):
    x = np.asarray(x)[~np.isnan(x)]
    return (np.percentile(x, 2.5), np.percentile(x, 97.5)) if len(x) else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", default=str(P1 / "rank_out.jsonl"))
    ap.add_argument("--ce", default=str(P1 / "ce_out.jsonl"))
    ap.add_argument("--outdir", default=str(R))
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. qwen -> table1_reversal_qwen.csv")
    ap.add_argument("--exclude-malformed", action="store_true", help="drop calls whose output failed to parse")
    ap.add_argument("--model-name", default="RankZephyr")
    ap.add_argument("--c5-perms", type=int, default=None,
                    help="use only C5 permutations 0..N-1 (time cut: qwen ran 3 perms; same cut for all models)")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    sfx = f"_{args.tag}" if args.tag else ""

    df = load(args.rank)
    if args.c5_perms is not None:
        df = df[~((df.cond == "C5") & (df.perm >= args.c5_perms))].copy()
    n_calls_all = len(df)
    if args.exclude_malformed:
        df = df[~df.malformed.astype(bool)].copy()
    u = reversal_table(df)
    boot = Boot(u.qid.unique())
    s, n = boot.mats(u, "rev")
    point = s.sum() / n.sum()
    bt = boot.rates(s, n)

    # Table 1: reversal and excess over C0
    t1 = []
    for c in CONDS:
        if c not in point:
            continue
        lo, hi = ci(bt[c])
        ex = bt[c] - bt["C0"]
        elo, ehi = ci(ex)
        t1.append(dict(cond=c, n_units=int((u.cond == c).sum()), reversal=point[c], ci_lo=lo, ci_hi=hi,
                       excess_vs_C0=point[c] - point["C0"], excess_ci_lo=elo, excess_ci_hi=ehi))
    t1 = pd.DataFrame(t1)
    calls = df[df.cond != "B"].groupby("cond")
    t1["malformed_rate"] = t1.cond.map(df.groupby("cond").malformed.mean()) if "malformed" in df else np.nan
    t1["prompt_tokens_mean"] = t1.cond.map(df.groupby("cond").prompt_tokens.mean()) if "prompt_tokens" in df else np.nan
    t1.to_csv(out / f"table1_reversal{sfx}.csv", index=False)

    # Table 2: direction among reversed units of unequal-grade pairs (+ supplementary C0-relative share)
    ue = u[u.unequal & (u.rev == 1.0)].copy()
    ue["opposite"] = (ue.after_correct == 0).astype(float)
    so, no = boot.mats(ue, "opposite")
    bo = boot.rates(so, no)
    t2 = []
    for c in CONDS:
        x = ue[ue.cond == c]
        if not len(x):
            continue
        k, m = int(x.opposite.sum()), len(x)
        p = binomtest(k, m, 0.5, alternative="greater").pvalue
        d = bo[c] - bo["C0"] if "C0" in bo else np.nan
        dlo, dhi = ci(d)
        t2.append(dict(cond=c, n_reversed=m, n_opposite=k, n_correct=m - k, opposite_share=k / m, binom_p=p,
                       opposite_minus_C0=k / m - ue[ue.cond == "C0"].opposite.mean(), diff_ci_lo=dlo, diff_ci_hi=dhi))
    t2 = pd.DataFrame(t2)
    t2.to_csv(out / f"table2_direction{sfx}.csv", index=False)

    # Table 3: strata
    u["gap_bin"] = pd.cut(u.gap0, [0, 2, 6, 19], labels=["<=2", "3-6", ">=7"])
    t3 = []
    for col, name in (("gap_bin", "base_rank_gap"), ("gdiff", "grade_diff")):
        g = (u.assign(pair_key=u.qid + "#" + u.pair_id.astype(str))
             .groupby(["cond", col], observed=True)
             .agg(reversal=("rev", "mean"), n_units=("rev", "size"), n_pairs=("pair_key", "nunique")).reset_index())
        g = g.rename(columns={col: "stratum"}); g.insert(0, "strata", name)
        t3.append(g)
    t3 = pd.concat(t3)
    t3.to_csv(out / f"table3_strata{sfx}.csv", index=False)

    # Supplementary: reversal split by whether pref0 agrees with the grade order (unequal pairs)
    sup = (u[u.unequal].groupby(["cond", "pref0_correct"]).rev.agg(["mean", "size"]).reset_index()
           .rename(columns={"mean": "reversal", "size": "n_units"}))
    base_acc = u[u.unequal].drop_duplicates(["qid", "pair_id"]).pref0_correct.mean()
    sup.to_csv(out / f"table_supp_pref0{sfx}.csv", index=False)

    # Control
    ce_rates = None
    if Path(args.ce).exists():
        dce = load(args.ce)
        uce = reversal_table(dce)
        ce_rates = uce.groupby("cond").rev.agg(["mean", "sum", "size"]).rename(columns={"mean": "reversal", "sum": "n_rev", "size": "n_units"})
        ce_rates.to_csv(out / f"table_control_ce{sfx}.csv")
    u.to_parquet(out / f"per_unit{sfx}.parquet")

    # Verdict
    t1i = t1.set_index("cond"); t2i = t2.set_index("cond")
    excess_pos = {c: bool(t1i.loc[c, "excess_ci_lo"] > 0) for c in TESTED if c in t1i.index}
    excess_has0 = {c: bool(t1i.loc[c, "excess_ci_lo"] <= 0 <= t1i.loc[c, "excess_ci_hi"]) for c in TESTED if c in t1i.index}
    dir_sig = {c: bool(c in t2i.index and t2i.loc[c, "binom_p"] < 0.05) for c in TESTED}
    ce_ok = ce_rates is not None and float(ce_rates.drop(index="R", errors="ignore").reversal.max()) == 0.0
    if not ce_ok:
        verdict = "PIPELINE_CHECK_FAILED" if ce_rates is not None else "CONTROL_MISSING"
    elif any(excess_pos[c] and dir_sig[c] for c in excess_pos):
        verdict = "GO"
    elif all(excess_has0.get(c, True) for c in ("C1", "C3", "C4")):
        verdict = "STOP"
    else:
        verdict = "INTERMEDIATE"

    cs = json.load(open(P1 / "logs" / "build_conditions.json"))
    bp = json.load(open(P1 / "logs" / "build_pairs.json"))
    with open(out / f"summary{sfx}.md", "w") as f:
        f.write(f"# Preference reversal under neighbor replacement ({args.model_name})\n\n")
        f.write(f"**Verdict (pre-specified rule): {verdict}**\n\n")
        f.write(f"Data: dl19+dl20, {u.qid.nunique()} queries (excluded {len(bp['excluded'])}), "
                f"{u[['qid','pair_id']].drop_duplicates().shape[0]} pairs, {len(df)} {args.model_name} calls"
                f"{f' (of {n_calls_all}; malformed excluded)' if args.exclude_malformed else ''}.\n\n")
        f.write("## Control (bge-reranker-v2-m3)\n\n")
        f.write((ce_rates.to_markdown() if ce_rates is not None else "missing") + "\n\n")
        f.write(f"Pipeline check (reversal 0 in every condition): **{'not run' if ce_rates is None else 'PASS' if ce_ok else 'FAIL'}**\n\n")
        f.write("## Table 1 — reversal rate and excess over C0 (95% query bootstrap CI)\n\n")
        f.write(t1.to_markdown(index=False, floatfmt=".4f") + "\n\n")
        f.write("## Table 2 — direction (unequal-grade pairs, reversed units)\n\n")
        f.write(t2.to_markdown(index=False, floatfmt=".4f") + "\n\n")
        f.write(f"Base-call accuracy on unequal pairs (pref0 agrees with grades): {base_acc:.3f}. "
                "Any reversal of a correct pref0 is 'opposite' by construction, so the opposite share mostly tracks "
                "this accuracy; `opposite_minus_C0` compares each condition with order-only shuffling (pre-specified supplementary analysis).\n\n")
        f.write("## Supplementary — reversal by pref0 correctness (unequal pairs)\n\n")
        f.write(sup.to_markdown(index=False, floatfmt=".4f") + "\n\n")
        f.write("## Table 3 — strata\n\n" + t3.to_markdown(index=False, floatfmt=".4f") + "\n\n")
        f.write("## Construction\n\n")
        f.write(f"- Excluded queries: {len(bp['excluded'])} {bp['excluded'] or ''}\n")
        f.write(f"- Base-set adjustments: {len(bp['adjust'])} queries; pair substitutions: {len(bp['pair_substitutions'])} queries "
                f"(details in logs/build_pairs.json)\n")
        f.write(f"- Calls with fewer than 18 replacements: " + ", ".join(f"{k} {v:.1%}" for k, v in cs['replace_lt18_frac'].items()) + "\n")
        f.write(f"- Mean replacements per call: " + ", ".join(f"{k} {v:.2f}" for k, v in cs['replace_mean'].items()) + "\n")
        f.write(f"- Length limit relaxed to ±40%: calls " + ", ".join(f"{k} {v:.1%}" for k, v in cs['tol40_call_frac'].items())
                + "; replacements " + ", ".join(f"{k} {v:.1%}" for k, v in cs['tol40_replacement_frac'].items()) + "\n")
        f.write("\n## Related work positioning\n\n"
                "- Frohmann et al. (arXiv:2608.26762): one pool document replaced, grade not controlled, list-level correlation. "
                "Here: all 18 neighbors replaced, grade controlled, pair reversal rate.\n"
                "- Chen et al. (arXiv:2409.16022): pointwise relevance grading. Here: listwise ranking.\n"
                "- Huang et al. (ACL 2026) study a setwise model; not cited as listwise.\n")
    print(open(out / f"summary{sfx}.md").read()[:3000])


if __name__ == "__main__":
    main()
