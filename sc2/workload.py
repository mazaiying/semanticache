"""
Workload construction for SemantiCache v2.

Source: LongBench multi-document QA tasks (hotpotqa, 2wikimqa, musique).
Each sample's passages form the document corpus. Every passage has a BASE
version that is materialized offline (chunk KV = materialized view).

For every sample we issue a short request stream:
  R0 orig    : base passages in original order
  R1 reorder : base passages, different retrieval order
  R2 format  : reordered; ~50% of passages are near-duplicates of their base
               (whitespace, title tag, sentence-boundary shift, neutral insert)
  R3 edit    : like R2, and the passage(s) holding the answer are FACT-EDITED:
               the answer span is replaced by a substitute; gold := substitute.
               (only for samples whose answer occurs verbatim in the context)
"""
from __future__ import annotations

import json
import random
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

INSTR = ("Answer the question based on the given passages. Only give me the "
         "answer and do not output any other words.")


# ------------------------------------------------------------------ metric
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = Counter(p) & Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    pr, rc = ns / len(p), ns / len(g)
    return 2 * pr * rc / (pr + rc)


def best_f1(pred: str, golds: List[str]) -> float:
    pred = pred.strip().split("\n")[0]
    return max(qa_f1(pred, g) for g in golds) if golds else 0.0


# ------------------------------------------------------------------ data
@dataclass
class Passage:
    pid: str          # stable id of the BASE document
    text: str
    version: str = "base"   # base | fmt_* | edit


@dataclass
class Request:
    rid: str
    sample_id: str
    kind: str                 # orig | reorder | format | edit
    question: str
    passages: List[Passage]
    golds: List[str]
    meta: Dict = field(default_factory=dict)


def split_passages(context: str) -> List[str]:
    parts = re.split(r"Passage \d+:\n", context)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def load_longbench(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f]


# ------------------------------------------------------------ perturbations
_SENT = re.compile(r"(?<=[.!?])\s+")


def perturb_format(text: str, rng: random.Random, neighbor: Optional[str]) -> (str, str):
    ops = ["ws", "title", "drop_first", "append_next", "insert"]
    op = rng.choice(ops)
    lines = text.split("\n", 1)
    if op == "ws":
        t = text.replace(". ", ".  ").replace("\n", "\n\n")
    elif op == "title":
        t = f"Title: {lines[0]}\n{lines[1]}" if len(lines) == 2 else f"Source: {text}"
    elif op == "drop_first":
        body = lines[1] if len(lines) == 2 else text
        sents = _SENT.split(body)
        if len(sents) > 2:
            body = " ".join(sents[1:])
        t = (lines[0] + "\n" + body) if len(lines) == 2 else body
    elif op == "append_next" and neighbor:
        ns = _SENT.split(neighbor.split("\n", 1)[-1])
        t = text + " " + ns[0]
    else:  # insert neutral sentence in the middle
        sents = _SENT.split(text)
        k = len(sents) // 2
        sents.insert(k, "This passage is reproduced from a public encyclopedia.")
        t = " ".join(sents)
        op = "insert"
    return t, "fmt_" + op


def perturb_drift(text: str, frac: float, rng: random.Random, donor: List[str]) -> str:
    """Content drift of magnitude `frac`: replace ~frac of the words with random
    words from a donor pool, in contiguous spans of 3-8 words (simulates
    document revisions of increasing size). Used for the maintenance-cost sweep."""
    words = text.split(" ")
    n = len(words)
    target = max(1, int(round(frac * n)))
    changed, tries = set(), 0
    while len(changed) < target and tries < 10000:
        L = rng.randint(3, 8)
        s = rng.randrange(0, max(1, n - L))
        for j in range(s, min(n, s + L)):
            if j not in changed:
                words[j] = rng.choice(donor)
                changed.add(j)
        tries += 1
    return " ".join(words)


def _is_numeric(a: str) -> bool:
    return bool(re.search(r"\d", a))


def build_requests(samples: List[dict], dataset: str, tokenizer, max_ctx_tokens: int,
                   n_samples: int, seed: int = 0, fmt_prob: float = 0.5,
                   drift_frac: float = 0.0, kinds: Optional[List[str]] = None
                   ) -> (List[Request], Dict[str, Passage]):
    """fmt_prob   : fraction of passages perturbed in `format`/`edit` requests
                    (default 0.5 reproduces the main experiments bit-for-bit).
       drift_frac : if > 0, add a `drift` request per sample: reordered base
                    passages where every passage NOT containing the answer has
                    ~drift_frac of its words rewritten (gold stays valid).
       kinds      : keep only these request kinds (None = all)."""
    rng = random.Random(seed)
    corpus: Dict[str, Passage] = {}
    pool_num = [s["answers"][0] for s in samples if _is_numeric(s["answers"][0])]
    pool_txt = [s["answers"][0] for s in samples
                if not _is_numeric(s["answers"][0]) and s["answers"][0].lower() not in ("yes", "no")]
    reqs: List[Request] = []
    used = 0
    for si, s in enumerate(samples):
        if used >= n_samples:
            break
        golds = s["answers"]
        ans = golds[0]
        if ans.lower() in ("yes", "no"):
            continue  # yes/no is too easy to guess; keep extractive answers
        texts = split_passages(s["context"])
        if len(texts) < 3:
            continue
        # token budget: keep passages in order until max_ctx, but always keep
        # passages containing the answer
        lens = [len(tokenizer(t, add_special_tokens=False).input_ids) for t in texts]
        has_ans = [ans.lower() in t.lower() for t in texts]
        keep, tot = [], 0
        for i, (t, L) in enumerate(zip(texts, lens)):
            if has_ans[i] or tot + L <= max_ctx_tokens:
                keep.append(i)
                tot += L
        keep = [i for i in keep]
        while tot > max_ctx_tokens * 1.15 and any(not has_ans[i] for i in keep):
            j = max((i for i in keep if not has_ans[i]), key=lambda i: lens[i])
            keep.remove(j)
            tot -= lens[j]
        sid = f"{dataset}-{si}"
        base = []
        for i in keep:
            p = Passage(f"{sid}-p{i}", texts[i])
            corpus[p.pid] = p
            base.append(p)
        q = s["input"]
        used += 1

        reqs.append(Request(f"{sid}-r0", sid, "orig", q, list(base), golds))
        order = list(base)
        rng.shuffle(order)
        reqs.append(Request(f"{sid}-r1", sid, "reorder", q, list(order), golds))

        def fmt_variant(ps):
            out = []
            for j, p in enumerate(ps):
                if rng.random() < fmt_prob:
                    nb = ps[j + 1].text if j + 1 < len(ps) else None
                    t, op = perturb_format(p.text, rng, nb)
                    out.append(Passage(p.pid, t, op))
                else:
                    out.append(p)
            return out

        order2 = list(base)
        rng.shuffle(order2)
        reqs.append(Request(f"{sid}-r2", sid, "format", q, fmt_variant(order2), golds))

        if any(has_ans[i] for i in keep):
            pool = pool_num if _is_numeric(ans) else pool_txt
            cands = [a for a in pool if normalize_answer(a) != normalize_answer(ans)
                     and a.lower() not in s["context"].lower()]
            if cands:
                sub = rng.choice(cands)
                order3 = list(base)
                rng.shuffle(order3)
                v3 = fmt_variant(order3)
                pat = re.compile(re.escape(ans), re.IGNORECASE)
                n_edit = 0
                for j, p in enumerate(v3):
                    if pat.search(p.text):
                        v3[j] = Passage(p.pid, pat.sub(sub, p.text), "edit")
                        n_edit += 1
                reqs.append(Request(f"{sid}-r3", sid, "edit", q, v3, [sub],
                                    {"orig_answer": ans, "n_edited": n_edit}))

        if drift_frac > 0:
            # separate RNG so the default request stream is unchanged
            drng = random.Random(seed * 100003 + si)
            donor = " ".join(texts).split(" ")
            order4 = list(base)
            drng.shuffle(order4)
            v4 = []
            for p in order4:
                if ans.lower() in p.text.lower():
                    v4.append(p)
                else:
                    v4.append(Passage(p.pid, perturb_drift(p.text, drift_frac, drng, donor), "drift"))
            reqs.append(Request(f"{sid}-r4", sid, "drift", q, v4, golds,
                                {"drift_frac": drift_frac}))
    if kinds:
        reqs = [r for r in reqs if r.kind in kinds]
    return reqs, corpus


# ---------------------------------------------------------- tokenization
class PromptBuilder:
    """Builds token ids as [prefix] + [passage_1] + ... + [suffix].
    Segments are tokenized independently so that a passage has identical
    tokens regardless of its position (needed for chunk reuse)."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        msgs = [{"role": "user", "content": "<<<SPLIT>>>"}]
        try:
            templ = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            templ = "<<<SPLIT>>>\n"
        head, tail = templ.split("<<<SPLIT>>>")
        self.prefix_text = head + INSTR + "\n\nThe following are given passages.\n"
        self.tail_text = tail
        self.prefix_ids = tokenizer(self.prefix_text, add_special_tokens=False).input_ids
        bos = tokenizer.bos_token_id
        if bos is not None and (not self.prefix_ids or self.prefix_ids[0] != bos) \
                and "<|im_start|>" not in head and getattr(tokenizer, "add_bos_token", True):
            self.prefix_ids = [bos] + self.prefix_ids
        self._cache: Dict[str, List[int]] = {}

    def passage_ids(self, text: str) -> List[int]:
        key = text
        if key not in self._cache:
            self._cache[key] = self.tok("\n\n" + text, add_special_tokens=False).input_ids
        return self._cache[key]

    def suffix_ids(self, question: str) -> List[int]:
        s = f"\n\n{INSTR}\n\nQuestion: {question}\nAnswer:" + self.tail_text
        return self.tok(s, add_special_tokens=False).input_ids
