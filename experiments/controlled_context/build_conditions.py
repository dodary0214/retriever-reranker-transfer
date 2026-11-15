"""One line per model call -> controlled/conditions.jsonl
fields: call_id, qid, pair_id, cond, rep, perm, doc_list (20 docids, slot order), n_replaced, tol (per-call max length
tolerance used: 0.2 / 0.4 / null when nothing was replaced), n_relaxed (replacements that needed the ±40% limit)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import COMMON, EXP
from build_pairs import rng_for, SLOT_A, SLOT_B
import numpy as np, pandas as pd

P1 = EXP / "controlled"
REPS = {"C0": 5, "C1": 3, "C2": 3, "C3": 1, "C4": 1}
N_PERM_C5 = 5
N_REPLICATE = 2
TA, TB = SLOT_A - 1, SLOT_B - 1  # 0-indexed target positions
NEIGH = [i for i in range(20) if i not in (TA, TB)]


def load_emb(ds):
    E = COMMON / "emb"
    ids = (E / f"{ds}.docs.ids.txt").read_text().split()
    qids = (E / f"{ds}.queries.ids.txt").read_text().split()
    return (dict(zip(ids, np.load(E / f"{ds}.docs.npy"))), dict(zip(qids, np.load(E / f"{ds}.queries.npy"))))


def pair_base(order, a, b):
    lst = list(order)
    ia, ib = lst.index(a), lst.index(b)
    lst[ia], lst[TA] = lst[TA], lst[ia]
    ib = lst.index(b)
    lst[ib], lst[TB] = lst[TB], lst[ib]
    assert lst[TA] == a and lst[TB] == b and len(set(lst)) == 20
    return lst


def length_ok(n0, n, tol):
    return abs(n - n0) <= tol * n0


def replace(base, grades, ntok_base, pool, mode, rng, score=None):
    """Replace the 18 neighbors in place. mode: same (C1), any (C2), sim (C3/C4, needs score dict)."""
    out, used, n_rep, tol_used, n_relax = list(base), set(base), 0, None, 0
    for i in NEIGH:
        d0, g0 = base[i], grades[i]
        n0 = ntok_base[d0]
        cands = pool[str(g0)] if mode in ("same", "sim") else {d: n for g in "0123" for d, n in pool[g].items()}
        for tol in (0.2, 0.4):
            ok = [d for d, n in cands.items() if d not in used and length_ok(n0, n, tol)]
            if ok:
                break
        if not ok:
            continue
        if mode == "sim":
            new = max(ok, key=lambda d: (score[d], d))
        else:
            new = ok[rng.integers(len(ok))]
        out[i] = new
        used.add(new)
        n_rep += 1
        n_relax += tol == 0.4
        tol_used = max(tol_used or 0, tol)
    return out, n_rep, (tol_used, n_relax)


def shuffle_neighbors(lst, rng):
    out = list(lst)
    vals = [lst[i] for i in NEIGH]
    for i, v in zip(NEIGH, rng.permutation(len(vals))):
        out[i] = vals[v]
    return out


def main():
    pairs = pd.read_csv(P1 / "pairs.tsv", sep="\t", dtype={"docA": str, "docB": str})
    base_sets = json.load(open(P1 / "base_sets.json"))
    emb = {ds: load_emb(ds) for ds in ("dl19", "dl20")}
    rows = []

    def add(r, cond, rep, perm, lst, n_rep=0, tol=(None, 0)):
        tol, n_relax = tol
        assert len(lst) == 20 and len(set(lst)) == 20 and lst[TA] == r.docA and lst[TB] == r.docB
        rows.append(dict(call_id=len(rows), qid=r.qid, pair_id=int(r.pair_id), cond=cond, rep=rep, perm=perm,
                         doc_list=lst, n_replaced=n_rep, tol=tol, n_relaxed=n_relax))

    for r in pairs.itertuples():
        ds, qid = r.qid.split(":")
        pool_f = json.load(open(P1 / "pools" / f"{r.qid.replace(':', '_')}.json"))
        pool, ntok_base = pool_f["pool"], pool_f["base_ntok"]
        bs = base_sets[r.qid]
        g_of = dict(zip(bs["order"], bs["grades"]))
        base = pair_base(bs["order"], r.docA, r.docB)
        grades = [g_of[d] for d in base]
        key = (r.qid, r.pair_id)
        add(r, "B", 0, 0, base)
        for k in range(N_REPLICATE):
            add(r, "R", k, 0, base)
        for k in range(REPS["C0"]):
            add(r, "C0", k, 0, shuffle_neighbors(base, rng_for(*key, "C0", k)))
        c1_lists = []
        for k in range(REPS["C1"]):
            lst, n, tol = replace(base, grades, ntok_base, pool, "same", rng_for(*key, "C1", k))
            add(r, "C1", k, 0, lst, n, tol)
            c1_lists.append((lst, n, tol))
        for k in range(REPS["C2"]):
            lst, n, tol = replace(base, grades, ntok_base, pool, "any", rng_for(*key, "C2", k))
            add(r, "C2", k, 0, lst, n, tol)
        D, Q = emb[ds]
        qv, av = Q[qid], D[r.docA]
        for cond, ref in (("C3", qv), ("C4", av)):
            score = {d: float(D[d] @ ref) for g in "0123" for d in pool[g]}
            lst, n, tol = replace(base, grades, ntok_base, pool, "sim", None, score)
            add(r, cond, 0, 0, lst, n, tol)
        for k, (lst, n, tol) in enumerate(c1_lists):
            for p in range(N_PERM_C5):
                add(r, "C5", k, p, shuffle_neighbors(lst, rng_for(*key, "C5", k, p)), n, tol)

    df = pd.DataFrame(rows)
    with open(P1 / "conditions.jsonl", "w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    rep = df[df.cond.isin(["C1", "C2", "C3", "C4"])]
    stats = {
        "n_calls": len(df), "calls_by_cond": df.cond.value_counts().sort_index().to_dict(),
        "replace_lt18_frac": rep.groupby("cond").n_replaced.apply(lambda s: float((s < 18).mean())).to_dict(),
        "replace_mean": rep.groupby("cond").n_replaced.mean().round(3).to_dict(),
        "tol40_call_frac": rep.groupby("cond").tol.apply(lambda s: float((s == 0.4).mean())).to_dict(),
        "tol40_replacement_frac": (rep.groupby("cond").n_relaxed.sum() / rep.groupby("cond").n_replaced.sum()).round(4).to_dict(),
    }
    json.dump(stats, open(P1 / "logs" / "build_conditions.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
