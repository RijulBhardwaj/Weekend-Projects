#!/usr/bin/env python3
"""Usage ledger + budget gate for Claude Code hooks.

  usage.py gate    PreToolUse: update ledger, block over-budget / expensive subagent spawns
  usage.py status  UserPromptSubmit: update ledger, print one status line into context
  usage.py record  Stop / SubagentStop: update ledger
  usage.py report  human-readable table (run by hand)
"""
import glob, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGET = os.path.join(ROOT, "usage", "budget.json")
LEDGER = os.path.join(ROOT, "usage", "ledger.json")
CHEAP_AGENTS = {"scout", "runner", "worker", "Explore", "web-fetch",
                "claude-code-guide", "statusline-setup"}
CHEAP_MODELS = {"haiku", "sonnet"}


def load(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def family(model):
    m = (model or "").lower()
    for fam in ("haiku", "sonnet", "opus", "fable", "mythos"):
        if fam in m:
            return "fable" if fam == "mythos" else fam
    return "fable"  # unknown: assume the most expensive


def cost(fam, u, cfg):
    w = cfg["weights_per_mtok"].get(fam, cfg["weights_per_mtok"]["fable"])
    tin = (u.get("input_tokens", 0)
           + u.get("cache_read_input_tokens", 0) * cfg.get("cache_read_factor", 0.1)
           + u.get("cache_creation_input_tokens", 0) * cfg.get("cache_write_factor", 1.25))
    return (tin * w["in"] + u.get("output_tokens", 0) * w["out"]) / 1e6


def update(hook, cfg):
    """Incrementally fold new transcript lines into the ledger."""
    led = load(LEDGER, {})
    sid = hook.get("session_id") or "unknown"
    s = led.setdefault(sid, {"files": {}, "models": {}, "total": 0.0})
    paths = set()
    tp = hook.get("transcript_path")
    if tp:
        paths.add(tp)
        paths.update(glob.glob(os.path.join(os.path.splitext(tp)[0], "**", "*.jsonl"),
                               recursive=True))
    if hook.get("agent_transcript_path"):
        paths.add(hook["agent_transcript_path"])
    for p in paths:
        st = s["files"].setdefault(p, {"off": 0, "last": None})
        try:
            with open(p, "rb") as f:
                f.seek(st["off"])
                data = f.read()
        except OSError:
            continue
        end = data.rfind(b"\n") + 1  # only complete lines
        for line in data[:end].splitlines():
            try:
                msg = json.loads(line).get("message") or {}
            except Exception:
                continue
            u = msg.get("usage")
            if not isinstance(u, dict) or msg.get("role") != "assistant":
                continue
            fam = family(msg.get("model"))
            if fam == "fable" and "synthetic" in (msg.get("model") or ""):
                continue
            c = cost(fam, u, cfg)
            toks = [u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                    + u.get("cache_creation_input_tokens", 0), u.get("output_tokens", 0)]
            m = s["models"].setdefault(fam, {"in": 0, "out": 0, "cost": 0.0, "msgs": 0})
            last = st["last"]
            if last and last["id"] == msg.get("id") and last["fam"] == fam:
                # same message re-logged (streamed chunks): replace, don't double count
                m["in"] -= last["t"][0]; m["out"] -= last["t"][1]
                m["cost"] -= last["c"]; m["msgs"] -= 1
            m["in"] += toks[0]; m["out"] += toks[1]; m["cost"] += c; m["msgs"] += 1
            st["last"] = {"id": msg.get("id"), "fam": fam, "t": toks, "c": c}
        st["off"] += end
    s["total"] = sum(m["cost"] for m in s["models"].values())
    tmp = LEDGER + ".tmp"
    with open(tmp, "w") as f:
        json.dump(led, f)
    os.replace(tmp, LEDGER)
    return s


def line(s, cfg):
    b = cfg["session_budget"]
    parts = ", ".join(f"{k} {v['cost']:.2f}" for k, v in sorted(s["models"].items()))
    return f"[usage] {s['total']:.2f}/{b:.2f} units ({100*s['total']/b:.0f}%) - {parts or 'none yet'}"


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "report"
    cfg = load(BUDGET, None)
    if cfg is None:
        sys.exit(0)
    if mode == "report":
        for sid, s in load(LEDGER, {}).items():
            print(sid, line(s, cfg))
            for k, v in sorted(s["models"].items()):
                print(f"  {k:7} msgs={v['msgs']:5} in={v['in']:>10} out={v['out']:>9} cost={v['cost']:.3f}")
        return
    try:
        hook = json.load(sys.stdin)
    except Exception:
        hook = {}
    s = update(hook, cfg)
    frac = s["total"] / cfg["session_budget"]
    if mode == "status":
        msg = line(s, cfg)
        if frac >= cfg.get("warn_at", 0.7):
            msg += " | NEAR LIMIT: haiku agents only, finish and summarise."
        print(msg)
    elif mode == "gate":
        if frac >= 1.0:
            sys.stderr.write(line(s, cfg) + "\nBUDGET EXHAUSTED: stop using tools, tell the user "
                             "what is done/pending. They can raise session_budget in "
                             ".claude/usage/budget.json.\n")
            sys.exit(2)
        if hook.get("tool_name") in ("Agent", "Task"):
            ti = hook.get("tool_input") or {}
            model, kind = ti.get("model"), ti.get("subagent_type") or "general-purpose"
            ok = model in CHEAP_MODELS or (model is None and kind in CHEAP_AGENTS)
            if frac >= cfg.get("warn_at", 0.7) and (model == "sonnet" or kind == "worker"):
                ok = False
            if not ok:
                sys.stderr.write(line(s, cfg) + "\nBlocked: subagents must be scout/runner (haiku) "
                                 "or worker (sonnet), or pass model haiku/sonnet. Above the warn "
                                 "threshold only haiku is allowed.\n")
                sys.exit(2)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # never break the session because of the tracker
        sys.stderr.write(f"usage.py error: {e}\n")
        sys.exit(0)
