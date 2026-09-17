#!/usr/bin/env python3
"""
Decision Output Paths v0.1 — a mechanics benchmark.

Three ways to get the same bounded decision out of one local model:

  generated   ask for JSON in free text, parse whatever comes back
  constrained decode into a JSON schema the server enforces
  direct      read the log-probabilities of the option tokens, decode nothing

It measures LATENCY, OUTPUT TOKENS, SCHEMA VALIDITY and AGREEMENT between the
paths. It does NOT measure whether the answers are correct, and it does NOT
measure calibration. Those need labelled data; this needs only a clock.

Everything is pinned and recorded: model digest, host, options, prompts, seed.
Raw per-call results are written as JSONL so anyone can recompute the summary.
"""

import argparse, json, statistics, sys, time, urllib.request
from datetime import datetime, timezone

OPTIONS = ["billing", "technical", "sales", "other"]

# 25 fixed inputs. Deliberately mixed: some obvious, some genuinely ambiguous.
# We are not scoring correctness, so ambiguity is fine and arguably useful.
INPUTS = [
    "My card was charged twice for the same order.",
    "The export button spins forever and never downloads anything.",
    "Do you offer volume pricing for 200 seats?",
    "I want to close my account and delete my data.",
    "Invoice 4021 shows tax at 20% but we are zero-rated.",
    "App crashes on launch since the last update, iPhone 15.",
    "Can someone walk our procurement team through the security review?",
    "Password reset email never arrives, checked spam.",
    "We were billed after cancelling in July.",
    "Is there an API rate limit on the starter plan?",
    "Your webhook stopped firing at 03:00 UTC.",
    "Renewal quote please, 12 months, same seat count.",
    "Two-factor codes are rejected even though the clock is right.",
    "The dashboard shows last month's numbers, not this month's.",
    "Where do I update the credit card on file?",
    "Hi.",
    "This is the third time I have written and nobody has replied.",
    "Data import fails with 'unexpected token' on a CSV that opens fine in Excel.",
    "What is the difference between the Pro and Business tiers?",
    "Refund requested, order placed by mistake 10 minutes ago.",
    "SSO with Okta returns a 500 after the assertion.",
    "Please send a W-9 for our records.",
    "Latency went from 200ms to 4s this morning.",
    "Do you have a student discount?",
    "Charged in USD but our contract says AUD.",
]

SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string", "enum": OPTIONS}},
    "required": ["category"],
}

INSTRUCTION = (
    "Classify the customer message into exactly one category.\n"
    f"Categories: {', '.join(OPTIONS)}.\n"
)


def post(host, path, payload, timeout=120):
    req = urllib.request.Request(
        host + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def call_generated(host, model, text):
    """Path 1: ask for JSON, parse whatever arrives."""
    prompt = (
        INSTRUCTION
        + 'Reply with only JSON of the form {"category": "..."}.\n\n'
        + f"Message: {text}\nJSON:"
    )
    t0 = time.perf_counter()
    d = post(host, "/api/generate", {
        "model": model, "prompt": prompt, "stream": False, "think": False,
        "options": {"temperature": 0, "num_predict": 32},
    })
    ms = (time.perf_counter() - t0) * 1000
    raw = (d.get("response") or "").strip()
    choice, valid = None, False
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        parsed = json.loads(raw[start:end])
        c = parsed.get("category")
        if c in OPTIONS:
            choice, valid = c, True
    except Exception:
        pass
    return {"ms": ms, "choice": choice, "schema_valid": valid,
            "output_tokens": d.get("eval_count"), "raw": raw[:200]}


def call_constrained(host, model, text):
    """Path 2: the server constrains decoding to the schema."""
    prompt = INSTRUCTION + f"\nMessage: {text}"
    t0 = time.perf_counter()
    d = post(host, "/api/generate", {
        "model": model, "prompt": prompt, "stream": False, "think": False,
        "format": SCHEMA,
        "options": {"temperature": 0, "num_predict": 32},
    })
    ms = (time.perf_counter() - t0) * 1000
    raw = (d.get("response") or "").strip()
    choice, valid = None, False
    try:
        c = json.loads(raw).get("category")
        if c in OPTIONS:
            choice, valid = c, True
    except Exception:
        pass
    return {"ms": ms, "choice": choice, "schema_valid": valid,
            "output_tokens": d.get("eval_count"), "raw": raw[:200]}


def call_direct(host, model, text):
    """
    Path 3: emit one token, read the distribution over the option tokens.

    The four labels have distinct first tokens, so the first position
    disambiguates all of them. We renormalise over just those, which is exactly
    what the option-scoring implementations do — and, as OpenJev's own README
    notes, that is a conditional distribution over the options shown, not a
    calibrated confidence.
    """
    prompt = (
        INSTRUCTION
        + "Answer with the single category word and nothing else.\n\n"
        + f"Message: {text}\nCategory:"
    )
    t0 = time.perf_counter()
    d = post(host, "/api/generate", {
        "model": model, "prompt": prompt, "stream": False, "think": False,
        "logprobs": True, "top_logprobs": 20,
        "options": {"temperature": 0, "num_predict": 1},
    })
    ms = (time.perf_counter() - t0) * 1000
    dist, lps = {}, (d.get("logprobs") or [])
    if lps:
        import math
        for cand in lps[0].get("top_logprobs", []):
            tok = (cand.get("token") or "").strip().lower()
            for opt in OPTIONS:
                if tok and opt.startswith(tok) and opt not in dist:
                    dist[opt] = math.exp(cand["logprob"])
    total = sum(dist.values())
    if total:
        dist = {k: v / total for k, v in dist.items()}
    choice = max(dist, key=dist.get) if dist else None
    return {"ms": ms, "choice": choice, "schema_valid": choice is not None,
            "output_tokens": d.get("eval_count"), "dist": dist,
            "covered": len(dist)}


PATHS = {"generated": call_generated, "constrained": call_constrained, "direct": call_direct}


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100)
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="qwen3:0.6b")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default="results.jsonl")
    a = ap.parse_args()

    info = post(a.host, "/api/show", {"model": a.model})
    digest = info.get("details", {}).get("parameter_size"), info.get("details", {}).get("quantization_level")
    print(f"model {a.model} {digest}", flush=True)

    for _ in range(a.warmup):
        for fn in PATHS.values():
            fn(a.host, a.model, INPUTS[0])
    print("warmup done", flush=True)

    rows = []
    with open(a.out, "w") as f:
        for rep in range(a.repeats):
            for i, text in enumerate(INPUTS):
                for name, fn in PATHS.items():
                    try:
                        r = fn(a.host, a.model, text)
                    except Exception as e:
                        r = {"ms": None, "choice": None, "schema_valid": False, "error": str(e)[:200]}
                    r.update({"path": name, "input_index": i, "repeat": rep})
                    rows.append(r)
                    f.write(json.dumps(r) + "\n")
            print(f"repeat {rep + 1}/{a.repeats} done", flush=True)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": a.model,
        "details": info.get("details", {}),
        "inputs": len(INPUTS),
        "repeats": a.repeats,
        "options": OPTIONS,
        "paths": {},
    }
    for name in PATHS:
        got = [r for r in rows if r["path"] == name and r.get("ms") is not None]
        lat = [r["ms"] for r in got]
        toks = [r["output_tokens"] for r in got if isinstance(r.get("output_tokens"), int)]
        summary["paths"][name] = {
            "n": len(got),
            "p50_ms": pct(lat, 50), "p95_ms": pct(lat, 95),
            "mean_ms": round(statistics.fmean(lat), 1) if lat else None,
            "median_output_tokens": statistics.median(toks) if toks else None,
            "schema_valid_rate": round(sum(1 for r in got if r["schema_valid"]) / len(got), 4) if got else None,
        }

    # Agreement: do the paths pick the same option for the same input?
    first = {}
    for r in rows:
        if r["repeat"] == 0:
            first.setdefault(r["input_index"], {})[r["path"]] = r["choice"]
    names = list(PATHS)
    agree = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a1, b1 = names[i], names[j]
            both = [v for v in first.values() if v.get(a1) and v.get(b1)]
            same = sum(1 for v in both if v[a1] == v[b1])
            agree[f"{a1}_vs_{b1}"] = {"compared": len(both),
                                      "agree": same,
                                      "rate": round(same / len(both), 4) if both else None}
    summary["agreement"] = agree

    print(json.dumps(summary, indent=2))
    with open(a.out.replace(".jsonl", "-summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
