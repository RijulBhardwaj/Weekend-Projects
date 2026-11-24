# Operating rules: minimal main-model usage

The main (expensive) model is an orchestrator only. Its job: understand the request,
split it, write tight specs, check results, give the final answer.

## Delegation
- Lookup, search, reading files/notebooks/docs -> `scout` (haiku).
- Running tests, installs, scripts, summarising logs -> `runner` (haiku).
- Writing or editing code, tests, docs -> `worker` (sonnet).
- Web pages -> `web-fetch`.
- Do it inline only when it is 1-2 tool calls on known paths, or the final judgement call.
- Never spawn `fork`, `general-purpose`, `Plan`, or any agent on opus/fable (the hook blocks it).
- Run independent agents in parallel in one message. Reuse a running agent via SendMessage
  instead of respawning.

## Specs to agents
Self-contained: goal, exact paths, constraints, what "done" means, and the report format
with a line cap. Agents start cold; include what you already know.

## Keeping main-model tokens low
- Do not read large files, notebooks or logs in the main thread; ask for excerpts.
- No planning documents, no restating, no re-verifying what an agent verified, unless risky.
- Short replies: result first, then only what the user needs.

## Budget
- `.claude/hooks/usage.py` tallies tokens per model from the transcripts into
  `.claude/usage/ledger.json`; a `[usage]` line is injected on each user prompt.
- Cap and weights: `.claude/usage/budget.json`. At `warn_at`: haiku agents only, wrap up.
  At 100%: all tools are blocked; report done/pending and stop. Only the user raises the cap.
- `python3 .claude/hooks/usage.py report` prints the table.

## Context hygiene (applies to every session)
- Never Read/cat files denied in settings or >50KB; use head/grep/offset via scout.
- Do not load MCP-heavy skills/plugins/connectors unless the task needs them; prefer ToolSearch over loading schemas.
- All agent reports are capped: scout 30, runner 30, worker 40, web-fetch 20 lines.
- No restating plans or summarising agent reports.
- End long threads with a fresh session rather than carrying 80k+ context.
