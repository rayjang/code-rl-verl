import json, re, collections, sys
P="/scratch/r919a03/code_verl_test/sources/t01_math_reward_rubric/rubrics/rubric_math.jsonl"
HANGUL=re.compile(u"[가-힣]")
rows=[json.loads(l) for l in open(P,encoding="utf-8") if l.strip()]
print("rows",len(rows))
top=collections.Counter(); crit_keys=collections.Counter(); tag_keys=collections.Counter()
ncrit=collections.Counter(); kinds=collections.Counter(); verify=collections.Counter(); rank=collections.Counter()
reward=collections.Counter(); inject=collections.Counter(); judge=collections.Counter(); points=collections.Counter()
lang=collections.Counter(); idx_prefix=collections.Counter(); kind_by_verify=collections.Counter()
rank_by_verify=collections.Counter(); pts_by_verify=collections.Counter(); crit_lang_mix=collections.Counter()
idxs=[]; per_row_kinds=collections.Counter(); per_row_verify_seq=collections.Counter(); per_row_rank_seq=collections.Counter()
per_row_pts_seq=collections.Counter(); uniq_crit=set(); vtypes=collections.Counter()
for r in rows:
    top.update(r.keys()); idxs.append(r["index"]); idx_prefix[r["index"].split("::")[0]]+=1
    cs=r["criteria"]; ncrit[len(cs)]+=1
    rk=set(); langs=set()
    for c in cs:
        crit_keys.update(c.keys()); t=c.get("tags",{}); tag_keys.update(t.keys())
        kinds[t.get("scaffold_kind")]+=1; verify[t.get("verify")]+=1; rank[t.get("scaffold_rank")]+=1
        reward[t.get("reward")]+=1; inject[t.get("inject")]+=1; judge[repr(t.get("judge"))]+=1; points[c.get("points")]+=1
        kind_by_verify[(t.get("scaffold_kind"),t.get("verify"))]+=1
        rank_by_verify[(t.get("verify"),t.get("scaffold_rank"))]+=1
        pts_by_verify[(t.get("verify"),c.get("points"))]+=1
        rk.add(t.get("scaffold_kind")); langs.add("ko" if HANGUL.search(c["criterion"]) else "en")
        uniq_crit.add(c["criterion"])
        for k,v in t.items(): vtypes[(k,type(v).__name__)]+=1
        vtypes[("points",type(c.get("points")).__name__)]+=1
    per_row_kinds[len(rk)]+=1
    per_row_verify_seq[tuple(c["tags"]["verify"] for c in cs)]+=1
    per_row_rank_seq[tuple(c["tags"]["scaffold_rank"] for c in cs)]+=1
    per_row_pts_seq[tuple(c["points"] for c in cs)]+=1
    crit_lang_mix[tuple(sorted(langs))]+=1
    lang["ko" if any(HANGUL.search(c["criterion"]) for c in cs) else "en"]+=1
print("top-level keys",top); print("criteria keys",crit_keys); print("tag keys",tag_keys)
print("criteria per row",ncrit); print("unique index",len(set(idxs)),"index prefix",idx_prefix)
print("scaffold_kind",kinds); print("verify",verify); print("scaffold_rank",rank); print("reward",reward); print("inject",inject); print("judge",judge); print("points",points)
print("value types",vtypes)
print("kind x verify",sorted(kind_by_verify.items())); print("verify x rank",sorted(rank_by_verify.items())); print("verify x points",sorted(pts_by_verify.items()))
print("distinct kinds per row",per_row_kinds); print("verify seq per row",per_row_verify_seq); print("rank seq per row",per_row_rank_seq); print("points seq per row",per_row_pts_seq)
print("row lang",lang); print("lang mix within row",crit_lang_mix); print("unique criterion strings",len(uniq_crit))
# per-kind language split
kl=collections.Counter()
for r in rows:
    for c in r["criteria"]:
        kl[(c["tags"]["scaffold_kind"],"ko" if HANGUL.search(c["criterion"]) else "en")]+=1
print("kind x lang",sorted(kl.items()))
# sample index lengths
print("index len",collections.Counter(len(i) for i in idxs))
# total points per row
print("total points per row",collections.Counter(sum(c["points"] for c in r["criteria"]) for r in rows))
# consecutive ko/en pattern check
pat=["ko" if any(HANGUL.search(c["criterion"]) for c in r["criteria"]) else "en" for r in rows]
print("first 12 langs",pat[:12])
print("alternating?",all(pat[i]!=pat[i+1] for i in range(len(pat)-1)))
