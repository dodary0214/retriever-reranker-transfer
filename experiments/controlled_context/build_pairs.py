"""Base candidate sets, target pairs, replacement pools.
out: controlled/pairs.tsv, controlled/base_sets.json, controlled/pools/<qid>.json, controlled/logs/build_pairs.json"""
import sys, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
from config import COMMON, EXP, SEED
import numpy as np, pandas as pd

P1 = EXP / "controlled"
N_BASE, MIN_JUDGED, SLOT_A, SLOT_B = 20, 60, 3, 12
UNEQUAL = [(3, 1), (3, 0), (2, 0), (2, 1), (3, 2), (1, 0)]


def rng_for(*key):
    h = hashlib.sha256(("|".join(map(str, (SEED,) + key))).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def base_counts(dist, avail):
    """Largest-remainder allocation of N_BASE over grades 0-3, then the plan's adjustments (a)-(c)."""
    raw = {g: N_BASE * dist.get(g, 0.0) for g in range(4)}
    cnt = {g: int(np.floor(v)) for g, v in raw.items()}
    for g in sorted(raw, key=lambda g: -(raw[g] - cnt[g]))[: N_BASE - sum(cnt.values())]:
        cnt[g] += 1
    log = []
    cnt = {g: min(c, avail.get(g, 0)) for g, c in cnt.items()}

    def take_from_largest(exclude):
        g = max((g for g in range(4) if g not in exclude and cnt[g] > 0), key=lambda g: cnt[g])
        cnt[g] -= 1
        return g

    def add(g, why, protect):
        if cnt[g] < avail.get(g, 0):
            src = take_from_largest(protect) if sum(cnt.values()) >= N_BASE else None
            cnt[g] += 1
            log.append(f"{why}: +1 grade{g} (from grade{src})")
            return True
        return False

    while cnt[2] + cnt[3] < 3:  # (a)
        if not (add(3, "a", {2, 3}) or add(2, "a", {2, 3})):
            break
    for g in range(4):  # (b)
        if cnt[g] == 0 and avail.get(g, 0) > 0:
            add(g, "b", {h for h in range(4) if cnt[h] <= 1} | {g})
    if cnt[2] < 2 and cnt[1] < 2:  # (c)
        g = 2 if avail.get(2, 0) >= 2 else 1
        add(g, "c", {h for h in range(4) if cnt[h] <= 1} | {g})
    while sum(cnt.values()) < N_BASE:  # fill if a grade ran out of judged docs
        g = max((g for g in range(4) if cnt[g] < avail.get(g, 0)), key=lambda g: avail[g] - cnt[g])
        cnt[g] += 1
        log.append(f"fill: +1 grade{g}")
    assert sum(cnt.values()) == N_BASE and cnt[2] + cnt[3] >= 3, cnt
    return cnt, log


def main():
    (P1 / "pools").mkdir(parents=True, exist_ok=True)
    (P1 / "logs").mkdir(parents=True, exist_ok=True)
    pairs_rows, base_sets, stats = [], {}, {"excluded": {}, "adjust": {}, "pair_substitutions": {}}
    for ds in ["dl19", "dl20"]:
        qrels = pd.read_csv(COMMON / "qrels" / f"{ds}.tsv", sep="\t", header=None, names=["qid", "docid", "grade"],
                            dtype={0: str, 1: str})
        run = pd.read_csv(COMMON / "runs" / f"{ds}.bm25.top100.txt", sep=" ", header=None, dtype={0: str, 2: str})
        run = run.rename(columns={0: "qid", 2: "docid"})[["qid", "docid"]]
        ntok = {json.loads(l)["docid"]: json.loads(l)["n_tokens"] for l in open(COMMON / "docs" / f"{ds}.jsonl")}
        for qid, qj in qrels.groupby("qid", sort=True):
            key = f"{ds}:{qid}"
            if len(qj) < MIN_JUDGED:
                stats["excluded"][key] = f"{len(qj)} judged docs < {MIN_JUDGED}"
                continue
            g_of = dict(zip(qj.docid, qj.grade))
            top = run[run.qid == qid].merge(qj, on=["qid", "docid"])
            dist = top.grade.value_counts(normalize=True).to_dict()
            avail = qj.grade.value_counts().to_dict()
            cnt, log = base_counts(dist, avail)
            if log:
                stats["adjust"][key] = log
            rng = rng_for(ds, qid, "base")
            base = []
            for g in range(4):
                docs = sorted(qj.docid[qj.grade == g])
                base += list(rng.choice(docs, size=cnt[g], replace=False)) if cnt[g] else []
            order = list(rng.permutation(base))
            # pairs
            by_g = {g: [d for d in order if g_of[d] == g] for g in range(4)}
            used, combos, subs = set(), [], []
            for c in UNEQUAL:
                if by_g[c[0]] and by_g[c[1]]:
                    combos.append(c)
                else:
                    subs.append(c)
            for c in subs:
                ok = [u for u in UNEQUAL if by_g[u[0]] and by_g[u[1]]]
                alt = min(ok, key=lambda u: (combos.count(u), ok.index(u)))  # least-used available combination
                combos.append(alt)
                stats["pair_substitutions"].setdefault(key, []).append(f"{c}->{alt}")
            eq = [(2, 2)] if len(by_g[2]) >= 2 else []
            eq += [(1, 1)] if len(by_g[1]) >= 2 else []
            if not eq:  # judged pool has neither two grade-2 nor two grade-1 docs
                eq = [(g, g) for g in (3, 0) if len(by_g[g]) >= 2][:2]
                stats["pair_substitutions"].setdefault(key, []).append(f"equal pairs: none of (2,2)/(1,1) -> {eq}")
            if len(eq) == 1:
                stats["pair_substitutions"].setdefault(key, []).append(f"equal pairs: only {eq[0]}")
                eq = eq * 2
            combos += eq[:2]
            prng = rng_for(ds, qid, "pairs")
            for pid, (ga, gb) in enumerate(combos):
                def pick(g, exclude):
                    cand = [d for d in by_g[g] if d not in used and d not in exclude] or \
                           [d for d in by_g[g] if d not in exclude]
                    return cand[prng.integers(len(cand))]
                a = pick(ga, set())
                b = pick(gb, {a})
                used |= {a, b}
                pairs_rows.append(dict(qid=key, pair_id=pid, docA=a, docB=b, gradeA=ga, gradeB=gb,
                                       slotA=SLOT_A, slotB=SLOT_B))
            base_sets[key] = {"order": order, "grades": [int(g_of[d]) for d in order]}
            # replacement pool: judged docs outside the base set, by grade, with token counts
            pool = {str(g): {d: ntok[d] for d in sorted(qj.docid[qj.grade == g]) if d not in set(order)} for g in range(4)}
            json.dump({"qid": key, "base_ntok": {d: ntok[d] for d in order}, "pool": pool},
                      open(P1 / "pools" / f"{key.replace(':', '_')}.json", "w"))
    pairs = pd.DataFrame(pairs_rows)
    pairs.to_csv(P1 / "pairs.tsv", sep="\t", index=False)
    json.dump(base_sets, open(P1 / "base_sets.json", "w"))
    stats.update(n_queries=len(base_sets), n_pairs=len(pairs),
                 combo_counts={f"{a},{b}": int(n) for (a, b), n in pairs.groupby(["gradeA", "gradeB"]).size().items()})
    json.dump(stats, open(P1 / "logs" / "build_pairs.json", "w"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) or len(v) < 8 else f"{len(v)} entries") for k, v in stats.items()}, indent=1))


if __name__ == "__main__":
    main()
