"""Compare zephyr / first / qwen on the same pairs and replacement documents.
in: models/per_unit_<model>.parquet (analyze.py --outdir models --tag <model>)
out: models/m1_reversal_by_model.csv, m2_direction_by_model.csv, m3_base_accuracy.csv, m4_gap_by_model.csv,
     r4_c3_excess_diff.csv, summary_models.md (R1-R4, pre-specified criteria)
All CIs: 95% query bootstrap, 1000 resamples, the same resampled query sets for every model (paired comparisons)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
from config import EXP, SEED
import numpy as np, pandas as pd

M = Path(sys.argv[1]) if len(sys.argv) > 1 else EXP / "controlled" / "models"  # output dir of analyze.py
MODELS = {"zephyr": "RankZephyr-7B (generative, trained)", "first": "FIRST-Mistral-7B (logits)",
          "qwen": "Qwen2.5-7B-Instruct (generative, zero-shot)"}
GENERATIVE = ["zephyr", "qwen"]
CONDS = ["R", "C0", "C1", "C2", "C3", "C4", "C5"]
N_BOOT = 1000


def ci(x):
    x = np.asarray(x)[~np.isnan(x)]
    return (np.percentile(x, 2.5), np.percentile(x, 97.5)) if len(x) else (np.nan, np.nan)


def boot(u, value, by, queries, idx, pair_mean):
    """Point rate and bootstrap matrix per column of `by`. pair_mean: average units within a pair first (table 1)."""
    if pair_mean:
        u = u.groupby(["qid", "pair_id", by])[value].mean().reset_index()
    s = u.pivot_table(index="qid", columns=by, values=value, aggfunc="sum").reindex(queries).fillna(0)
    n = u.pivot_table(index="qid", columns=by, values=value, aggfunc="count").reindex(queries).fillna(0)
    S, N = s.values[idx].sum(1), n.values[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s.sum() / n.sum(), pd.DataFrame(S / N, columns=s.columns)


def main():
    U = {m: pd.read_parquet(M / f"per_unit_{m}.parquet") for m in MODELS if (M / f"per_unit_{m}.parquet").exists()}
    queries = sorted(set().union(*(u.qid.unique() for u in U.values())))
    idx = np.random.default_rng(SEED).integers(0, len(queries), size=(N_BOOT, len(queries)))
    m1, m2, m3, m4, exc_boot = [], [], [], [], {}

    for m, u in U.items():
        p, b = boot(u, "rev", "cond", queries, idx, True)
        exc_boot[m] = b["C3"] - b["C0"]
        for c in CONDS:
            lo, hi = ci(b[c]); elo, ehi = ci(b[c] - b["C0"])
            m1.append(dict(model=m, cond=c, n_units=int((u.cond == c).sum()), reversal=p[c], ci_lo=lo, ci_hi=hi,
                           excess_vs_C0=p[c] - p["C0"], excess_ci_lo=elo, excess_ci_hi=ehi))
        ue = u[u.unequal & (u.rev == 1.0)].assign(opposite=lambda d: (d.after_correct == 0).astype(float))
        p, b = boot(ue, "opposite", "cond", queries, idx, False)
        for c in CONDS:
            if c not in p or np.isnan(p[c]):
                continue
            lo, hi = ci(b[c])
            m2.append(dict(model=m, cond=c, n_reversed=int((ue.cond == c).sum()), opposite_share=p[c], ci_lo=lo, ci_hi=hi))
        v = u[u.unequal]
        acc = v.drop_duplicates(["qid", "pair_id"]).pref0_correct.mean()
        v = v.assign(grp=v.cond + "|" + np.where(v.pref0_correct == 1, "correct", "wrong"))
        p, b = boot(v, "rev", "grp", queries, idx, True)
        for c in CONDS:
            row = dict(model=m, cond=c, base_accuracy=acc)
            for g in ("wrong", "correct"):
                k = f"{c}|{g}"
                lo, hi = ci(b[k])
                row.update({f"rev_base_{g}": p[k], f"rev_base_{g}_ci_lo": lo, f"rev_base_{g}_ci_hi": hi,
                            f"n_base_{g}": int((v.grp == k).sum())})
            m3.append(row)
        u = u.assign(gap_bin=pd.cut(u.gap0, [0, 2, 6, 19], labels=["<=2", "3-6", ">=7"]))
        g = u.groupby(["cond", "gap_bin"], observed=True).rev.agg(["mean", "size"]).reset_index()
        m4.append(g.rename(columns={"mean": "reversal", "size": "n_units"}).assign(model=m))

    m1, m2, m3 = pd.DataFrame(m1), pd.DataFrame(m2), pd.DataFrame(m3)
    m4 = pd.concat(m4)[["model", "cond", "gap_bin", "reversal", "n_units"]]
    m1.to_csv(M / "m1_reversal_by_model.csv", index=False)
    m2.to_csv(M / "m2_direction_by_model.csv", index=False)
    m3.to_csv(M / "m3_base_accuracy.csv", index=False)
    m4.to_csv(M / "m4_gap_by_model.csv", index=False)

    # R4: C3 excess of each generative model minus FIRST's, on the same bootstrap samples
    r4 = []
    if "first" in exc_boot:
        f1 = m1.set_index(["model", "cond"])
        for g in [x for x in GENERATIVE if x in exc_boot]:
            d = exc_boot[g] - exc_boot["first"]
            lo, hi = ci(d)
            a, b_ = f1.loc[(g, "C3")], f1.loc[("first", "C3")]
            r4.append(dict(generative=g, diff_c3_excess=a.excess_vs_C0 - b_.excess_vs_C0, diff_ci_lo=lo, diff_ci_hi=hi,
                           ci_non_overlap=bool(a.excess_ci_lo > b_.excess_ci_hi)))
    r4 = pd.DataFrame(r4)
    r4.to_csv(M / "r4_c3_excess_diff.csv", index=False)

    # R1-R3 per model
    f1 = m1.set_index(["model", "cond"]); f2 = m2.set_index(["model", "cond"])
    rows = []
    for m in U:
        r0 = f1.loc[(m, "R"), "reversal"]
        R1 = all(f1.loc[(m, c), "excess_ci_lo"] <= 0 <= f1.loc[(m, c), "excess_ci_hi"] for c in ("C1", "C4"))
        R2 = not (f1.loc[(m, "C3"), "excess_ci_lo"] <= 0 <= f1.loc[(m, "C3"), "excess_ci_hi"])
        r3c = {c: bool(f2.loc[(m, c), "ci_lo"] <= 0.5 <= f2.loc[(m, c), "ci_hi"]) for c in ("C1", "C2", "C3", "C4", "C5")
               if (m, c) in f2.index}
        rows.append(dict(model=m, R_reversal=r0, R1=R1, R2=R2, R3=all(r3c.values()),
                         R3_failing=",".join(c for c, ok in r3c.items() if not ok)))
    verdict = pd.DataFrame(rows)
    verdict.to_csv(M / "r123_by_model.csv", index=False)
    general = bool(len(verdict) == 3 and verdict[["R1", "R2", "R3"]].all().all())

    with open(M / "summary_models.md", "w") as f:
        f.write("# Controlled candidate context — listwise rerankers on the same pairs\n\n")
        f.write("Models: " + "; ".join(f"`{m}` = {MODELS[m]}" for m in U) + "\n\n")
        f.write("## R1–R3 per model (criteria fixed before results)\n\n" + verdict.to_markdown(index=False) + "\n\n")
        f.write(f"**Generalization (R1–R3 in all three models): {'YES → claim listwise LLM rerankers in general' if general else 'NO → report per model'}**\n\n")
        f.write("## R4 — C3 excess, generative minus FIRST (paired query bootstrap)\n\n"
                + (r4.to_markdown(index=False, floatfmt=".4f") if len(r4) else "n/a") + "\n\n")
        f.write("## m1 — reversal and excess over C0\n\n" + m1.to_markdown(index=False, floatfmt=".4f") + "\n\n")
        f.write("## m2 — opposite share (unequal-grade reversals)\n\n" + m2.to_markdown(index=False, floatfmt=".4f") + "\n\n")
        f.write("## m3 — base accuracy and reversal by base correctness\n\n" + m3.to_markdown(index=False, floatfmt=".4f") + "\n\n")
        f.write("## m4 — reversal by base rank gap\n\n"
                + m4.pivot_table(index=["model", "gap_bin"], columns="cond", values="reversal", observed=True)
                .to_markdown(floatfmt=".3f") + "\n")
    print(open(M / "summary_models.md").read()[:4000])


if __name__ == "__main__":
    main()
