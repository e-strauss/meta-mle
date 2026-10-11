import "server-only";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";
import { argValue, cmdline, lockHolder } from "../procs";
import type {
  Adapter, Budget, ContextSizes, FileRef, GraphEdge, GraphNode, NodeStatus, RunDetail, RunState, RunSummary,
  ScorePoint, Section, TimelineEvent,
} from "../types";

// Reads a meta-mle workspace (state.db + artifacts/) strictly read-only.
// This file is the only place in the dashboard that knows meta-mle's layout.

type Rec = Record<string, any>;

// A locked contract is immutable and can hold exact fold memberships (millions of row
// positions on large tasks). Load it once per workspace, without its splits, and
// afterwards only check that it still exists, which never reads the payload.
const contracts = new Map<string, Rec>();

function lockedContract(db: DatabaseSync, dir: string): Rec | undefined {
  if (!db.prepare("SELECT 1 FROM meta WHERE key = 'contract'").get()) return undefined;
  if (!contracts.has(dir)) {
    const row = db.prepare("SELECT json_remove(payload, '$.splits') AS payload FROM meta WHERE key = 'contract'")
      .get() as Rec;
    contracts.set(dir, JSON.parse(row.payload));
  }
  return contracts.get(dir);
}

const setups = new Map<string, Rec>();

function setupRecord(db: DatabaseSync, dir: string, id: string): Rec {
  const key = `${dir}\0${id}`;
  if (setups.has(key)) return { ...setups.get(key)! };
  const row = db.prepare("SELECT json_remove(payload, '$.result.snapshot.splits') AS payload FROM records WHERE id = ?")
    .get(id) as Rec;
  const rec = JSON.parse(row.payload);
  if (rec.status !== "running") setups.set(key, rec); // finished setups never change
  return { ...rec };
}

class Workspace {
  meta: Record<string, any> = {};
  records: Map<string, Rec[]> = new Map();
  byId: Map<string, Rec> = new Map();
  events: { time: number; kind: string; payload: Rec }[] = [];

  constructor(public dir: string) {
    const db = new DatabaseSync(path.join(dir, "state.db"), { readOnly: true, timeout: 2000 });
    try {
      for (const row of db.prepare("SELECT key, payload FROM meta WHERE key != 'contract'").all() as Rec[]) {
        this.meta[row.key] = JSON.parse(row.payload);
      }
      this.meta.contract = lockedContract(db, dir);
      // Setup records repeat the fold memberships in their snapshot; same treatment.
      const rows = db.prepare(
        "SELECT kind, id, CASE WHEN kind = 'evaluation_setup' THEN NULL ELSE payload END AS payload FROM records ORDER BY seq",
      ).all() as Rec[];
      for (const row of rows) {
        const rec = row.payload === null ? setupRecord(db, dir, row.id) : JSON.parse(row.payload);
        rec._seq = this.byId.size;
        if (!this.records.has(row.kind)) this.records.set(row.kind, []);
        this.records.get(row.kind)!.push(rec);
        this.byId.set(rec.id, rec);
      }
      for (const row of db.prepare("SELECT time, kind, payload FROM events ORDER BY seq").all() as Rec[]) {
        this.events.push({ time: row.time * 1000, kind: row.kind, payload: JSON.parse(row.payload) });
      }
    } finally {
      db.close();
    }
  }

  of(kind: string): Rec[] {
    return this.records.get(kind) ?? [];
  }
}

function readJson(file: string): any {
  try {
    return JSON.parse(readFileSync(file, "utf8"));
  } catch {
    return undefined;
  }
}

function readText(file: string, max = 200_000): string | undefined {
  try {
    const text = readFileSync(file, "utf8");
    return text.length > max ? text.slice(0, max) + `\n… truncated (${text.length} chars)` : text;
  } catch {
    return undefined;
  }
}

function listFiles(dir: string, base: string): FileRef[] {
  if (!existsSync(dir)) return [];
  const out: FileRef[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.name === "mpl-cache") continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...listFiles(full, base));
    else out.push({ path: path.relative(base, full), size: statSync(full).size });
  }
  return out.sort((a, b) => a.path.localeCompare(b.path));
}

function status(s: string | undefined): NodeStatus {
  if (s === "ok" || s === "failed" || s === "running" || s === "rejected" || s === "interrupted") return s;
  return "info";
}

function short(id: string): string {
  return id.replace(/^(expansion|exploration|setup)_/, "").replace(/_variant_/, " v");
}

function fmt(x: unknown): string {
  if (x === null || x === undefined) return "—";
  if (typeof x === "number") return Number.isInteger(x) ? String(x) : x.toPrecision(5);
  if (typeof x === "string") return x;
  return JSON.stringify(x);
}

// sklearn reports mean seconds per fold; show it with the total over folds, which
// is what the attempt's time breakdown (grid_search.fold_fit / fold_score) adds up.
function perFold(label: string, mean: number | undefined, folds: number | undefined): [string, string][] {
  if (typeof mean !== "number") return [[label, "—"]];
  const total = folds ? ` · ${(mean * folds).toFixed(1)}s total over ${folds} folds` : "";
  return [[label, `${mean.toFixed(1)}s per fold${total}`]];
}

function counts(ws: Workspace) {
  const attempts = ws.of("attempt");
  return {
    model_calls: ws.of("model_call").length,
    actions: Number(ws.meta.actions ?? 0),
    explorations: ws.of("exploration").length,
    evaluation_setups: ws.of("evaluation_setup").length,
    expansions: ws.of("expansion").length,
    probes: ws.of("probe").length,
    evaluations: attempts.reduce((n, a) => n + (a.evaluation_count ?? 0), 0),
    repairs: attempts.filter((a) => (a.repair_number ?? 0) > 0).length,
  };
}

function summarize(ws: Workspace, id: string): RunSummary {
  const budget = ws.meta.budget ?? {};
  const c = counts(ws);
  const budgets: Budget[] = [
    { name: "model calls", used: c.model_calls, limit: budget.max_model_calls ?? null },
    { name: "actions", used: c.actions, limit: budget.max_actions ?? null },
    { name: "explorations", used: c.explorations, limit: budget.max_explorations ?? null },
    { name: "setups", used: c.evaluation_setups, limit: budget.max_evaluation_setups ?? null },
    { name: "expansions", used: c.expansions, limit: budget.max_expansions ?? null },
    { name: "evaluations", used: c.evaluations, limit: budget.max_evaluations ?? null },
    { name: "probes", used: c.probes, limit: budget.max_probes ?? null },
    { name: "repairs", used: c.repairs, limit: null },
    { name: "minutes", used: Math.round(Number(ws.meta.elapsed_s ?? 0) / 60),
      limit: budget.max_wall_seconds ? Math.round(budget.max_wall_seconds / 60) : null },
  ];
  const valid = ws.of("candidate").filter((x) => x.status === "ok" && typeof x.score === "number");
  const best = valid.reduce<Rec | undefined>((b, x) => (!b || x.score > b.score ? x : b), undefined);
  const pid = lockHolder(path.join(ws.dir, ".run.lock"));
  const state = (ws.meta.state ?? "unknown") as RunState;
  const notes: string[] = [];
  if (state === "running" && !pid) notes.push("State is 'running' but no process holds the lock (killed?)");
  if (ws.meta.task && !ws.meta.contract && ws.of("candidate").length === 0 && state === "complete") {
    notes.push("Completed without a locked evaluation");
  }
  const desc: string = ws.meta.task?.description ?? "";
  return {
    id,
    adapter: "meta-mle",
    title: desc.split("\n").find((l) => l.trim())?.replace(/^#+\s*/, "").slice(0, 120) ?? id,
    state,
    live: Boolean(pid),
    pid,
    policy: ws.meta.policy,
    model: ws.meta.model ?? (pid ? argValue(cmdline(pid), "--model") : undefined),
    metric: ws.meta.contract?.spec?.scoring ?? ws.meta.contract?.scoring,
    best: best ? { score: best.score, node: best.id } : undefined,
    budgets,
    createdAt: ws.events[0]?.time,
    updatedAt: ws.events.at(-1)?.time,
    notes,
  };
}

function graph(ws: Workspace): { nodes: GraphNode[]; edges: GraphEdge[] } {
  const contract = ws.meta.contract;
  const nodes: GraphNode[] = [{
    id: "root", kind: "root", shape: "square", label: "root",
    subtitle: contract ? `${contract.spec?.scoring ?? ""} · ${contract.rows} rows` : "no evaluation lock",
    status: contract ? "ok" : "info", parent: null, order: -1,
  }];
  const edges: GraphEdge[] = [];
  const candidates = ws.of("candidate");
  const byBatch = new Map<string, Rec[]>();
  for (const c of candidates) {
    if (!byBatch.has(c.batch_id)) byBatch.set(c.batch_id, []);
    byBatch.get(c.batch_id)!.push(c);
  }

  for (const s of ws.of("evaluation_setup")) {
    nodes.push({
      id: s.id, kind: "setup", shape: "square", label: "setup " + short(s.id),
      subtitle: `${(s.attempt_ids ?? []).length} attempt(s)`, status: status(s.status),
      parent: "root", order: s._seq,
    });
    edges.push({ from: "root", to: s.id, kind: "requested" });
  }

  for (const e of ws.of("expansion")) {
    const batch = byBatch.get(e.id) ?? [];
    if (batch.length === 0) {
      // Running, rejected (contract drift) or failed before producing variants.
      nodes.push({
        id: e.id, kind: "expansion", shape: "circle", label: "expansion " + short(e.id),
        subtitle: e.status === "running" ? "in progress" : (e.error ?? e.result?.error ?? "").slice(0, 60),
        status: status(e.status), parent: e.parent_id ?? "root", order: e._seq,
      });
      edges.push({ from: e.parent_id ?? "root", to: e.id, kind: "primary" });
    }
  }

  for (const c of candidates) {
    const config = c.configuration_description && Object.keys(c.configuration_description).length
      ? Object.values(c.configuration_description).join(", ") : undefined;
    nodes.push({
      id: c.id, kind: "candidate", shape: "circle", label: short(c.id),
      subtitle: config ?? (c.description ?? "").slice(0, 60),
      status: status(c.status), score: c.score ?? null, parent: c.parent_id ?? "root",
      group: (byBatch.get(c.batch_id)?.length ?? 0) > 1 ? c.batch_id : undefined, order: c._seq,
    });
    edges.push({ from: c.parent_id ?? "root", to: c.id, kind: "primary" });
    for (const r of c.reference_ids ?? []) edges.push({ from: r, to: c.id, kind: "reference" });
  }

  for (const x of ws.of("exploration")) {
    const expansion = x.expansion_id ? ws.byId.get(x.expansion_id) : undefined;
    const anchor = expansion?.parent_id ?? "root";
    nodes.push({
      id: x.id, kind: "exploration", shape: "diamond", label: "explore " + short(x.id),
      subtitle: (x.question ?? "").slice(0, 60), status: status(x.status), parent: anchor, order: x._seq,
    });
    edges.push({ from: anchor, to: x.id, kind: "requested" });
    if (expansion) {
      const targets = byBatch.get(expansion.id)?.map((c) => c.id) ?? [expansion.id];
      for (const t of targets) edges.push({ from: x.id, to: t, kind: "evidence" });
    }
  }
  for (const pr of ws.of("probe")) {
    // Out-of-fold predictions of a candidate (or of a new pipeline): evidence, no score node.
    const anchor = pr.candidate_id && ws.byId.has(pr.candidate_id) ? pr.candidate_id : "root";
    const info = pr.result?.probe;
    nodes.push({
      id: pr.id, kind: "probe", shape: "diamond", label: "probe " + short(pr.id.replace(/^probe_/, "")),
      subtitle: info ? `oof ${info.score?.toFixed?.(4) ?? ""} · ${(pr.question ?? "").slice(0, 40)}` : (pr.question ?? "").slice(0, 60),
      status: status(pr.status), parent: anchor, order: pr._seq,
    });
    edges.push({ from: anchor, to: pr.id, kind: "requested" });
  }
  return { nodes, edges };
}

function eventText(kind: string, p: Rec): string {
  switch (kind) {
    case "controller_decision":
      return `${p.action}: ${p.question ?? p.reason ?? ""}`;
    case "execution_finished":
      return `${p.owner_id} ${p.status}${p.evaluations ? ` (${p.evaluations} evals)` : ""}`;
    case "model_call_started":
    case "model_call_finished":
      return `${p.method} ${p.id}`;
    default:
      return Object.entries(p).map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`).join(" ");
  }
}

// Worker phase timings (response.json, or timings.json while running / after a kill).
// Rows nested under grid_search are a breakdown of it, not additional time.
function timingSection(response: Rec | undefined, live: Rec | undefined, writtenAt?: number): Section | undefined {
  const t = response?.timings ?? live;
  if (!t?.phases) return undefined;
  const wall = response?.wall_s as number | undefined;
  const total = wall ?? t.elapsed_s;
  const pct = (x: number) => (total ? ` (${((100 * x) / total).toFixed(0)}%)` : "");
  const rows: [string, string][] = t.phases.map((p: Rec) => {
    const what = p.output ? ` · ${p.output}` : p.variants ? ` · ${p.variants} variants × ${p.folds} folds` : "";
    return [`${p.part_of ? "  └ " : ""}${p.phase}${what}`, `${p.seconds.toFixed(2)}s${pct(p.seconds)}`];
  });
  const counted = t.phases.filter((p: Rec) => !p.part_of).reduce((n: number, p: Rec) => n + p.seconds, 0);
  if (wall !== undefined) {
    rows.push(["startup + unaccounted", `${Math.max(0, wall - counted).toFixed(2)}s${pct(Math.max(0, wall - counted))}`]);
    rows.push(["wall", `${wall.toFixed(2)}s`]);
  }
  // timings.json is rewritten when a phase starts, so its mtime dates the current phase.
  if (t.in_progress && response) {
    rows.push(["killed during", t.in_progress]);
  } else if (t.in_progress) {
    const since = writtenAt ? ` for ${((Date.now() - writtenAt) / 1000).toFixed(0)}s` : "";
    rows.push(["in progress", `${t.in_progress}${since}`]);
  }
  return { title: "time breakdown", kind: "kv", content: rows };
}

// Top-level drawing for a node: its last successful attempt, else its last one.
function graphSection(ws: Workspace, ids: string[]): Section[] {
  const attempts = ids.map((id) => ws.byId.get(id)).filter(Boolean) as Rec[];
  const pick = [...attempts].reverse().find((a) => a.status === "ok") ?? attempts.at(-1);
  if (!pick || !existsSync(path.join(ws.dir, pick.path, "plan.py"))) return [];
  return [{ title: `Skrub DataOps graph (attempt ${pick.repair_number ?? 0}, ${pick.status})`, kind: "graph",
    content: pick.path }];
}

function attemptSections(ws: Workspace, ids: string[]): Section[] {
  return ids.map((id) => {
    const a = ws.byId.get(id);
    if (!a) return { title: id, kind: "text", content: "missing attempt record" } as Section;
    const dir = path.join(ws.dir, a.path);
    const response = readJson(path.join(dir, "response.json"));
    const inner: Section[] = [{
      title: "status", kind: "kv", content: [
        ["status", fmt(a.status)], ["repair", fmt(a.repair_number)],
        ["evaluations charged", fmt(a.evaluation_count)],
        ["duration (s)", fmt(response?.duration_s)],
        ...(a.warning ? [["warning", fmt(a.warning)] as [string, string]] : []),
      ],
    }];
    const timingsFile = path.join(dir, "timings.json");
    const timing = timingSection(response, readJson(timingsFile),
      existsSync(timingsFile) ? statSync(timingsFile).mtimeMs : undefined);
    if (timing) inner.push(timing);
    if (response?.error) inner.push({ title: "error", kind: "code", content: response.traceback ?? response.error });
    const plan = readText(path.join(dir, "plan.py"));
    if (plan) {
      inner.push({ title: "Skrub DataOps graph", kind: "graph", content: a.path });
      inner.push({ title: "plan.py", kind: "code", content: plan, lang: "python" });
    }
    inner.push({ title: "files", kind: "files", content: listFiles(dir, ws.dir) });
    return { title: `attempt ${a.repair_number ?? ""} · ${a.status}`, kind: "group", content: inner } as Section;
  });
}

function nodeSections(ws: Workspace, nodeId: string): Section[] {
  const meta = ws.meta;
  if (nodeId === "root") {
    const c = meta.contract;
    const out: Section[] = [{ title: "task", kind: "text", content: meta.task?.description ?? "" }];
    if (c) {
      out.push({ title: "evaluation contract", kind: "kv", content: [
        ["id", fmt(c.id)], ["scoring", fmt(c.spec?.scoring ?? c.scoring)], ["rows", fmt(c.rows)],
        ["folds", fmt(c.audit?.folds)], ["fold fingerprint", fmt(c.fold_fingerprint)],
      ] });
      if (c.audit) out.push({ title: "audit", kind: "json", content: c.audit });
      if (c.spec) out.push({ title: "spec", kind: "json", content: c.spec });
      if (c.setup_source) out.push({ title: "locked evaluation source", kind: "code", content: c.setup_source, lang: "python" });
    }
    out.push({ title: "sources", kind: "json", content: meta.sources });
    return out;
  }
  const rec = ws.byId.get(nodeId);
  if (!rec) return [{ title: "not found", kind: "text", content: nodeId }];

  if (nodeId.includes("_variant_")) {
    const expansion = ws.byId.get(rec.batch_id) ?? {};
    const out: Section[] = [{ title: "candidate", kind: "kv", content: [
      ["status", fmt(rec.status)], ["score", fmt(rec.score)], ["std", fmt(rec.std)],
      ["fold scores", (rec.fold_scores ?? []).map(fmt).join(" / ") || "—"],
      ...perFold("fit time", rec.mean_fit_time, rec.fold_scores?.length),
      ...perFold("score time", rec.mean_score_time, rec.fold_scores?.length),
      ["parent", fmt(rec.parent_id)], ["references", (rec.reference_ids ?? []).join(", ") || "—"],
      ["batch", fmt(rec.batch_id)],
    ] }];
    if (rec.error) out.push({ title: "error", kind: "code", content: rec.error });
    if (rec.configuration_description && Object.keys(rec.configuration_description).length) {
      out.push({ title: "configuration", kind: "json", content: rec.configuration_description });
    }
    if (expansion.proposal) out.push({ title: "proposal", kind: "json", content: expansion.proposal });
    out.push(...graphSection(ws, expansion.attempt_ids ?? []));
    out.push(...findingSections(ws, rec.finding_ids ?? [], "findings available to the planner"));
    out.push({ title: "attempts (repair chain)", kind: "group", content: attemptSections(ws, expansion.attempt_ids ?? []) });
    return out;
  }
  if (nodeId.startsWith("exploration_")) {
    const out: Section[] = [{ title: "exploration", kind: "kv", content: [
      ["status", fmt(rec.status)], ["question", fmt(rec.question)],
      ["stopping condition", fmt(rec.stopping_condition)],
      ["requested by", rec.expansion_id ?? "controller"],
    ] }];
    const findings = ws.of("finding").filter((f) => f.exploration_id === nodeId).map((f) => f.id);
    out.push(...findingSections(ws, findings, "findings"));
    out.push(...graphSection(ws, rec.attempt_ids ?? []));
    const outputs = rec.result?.outputs as Record<string, Rec> | undefined;
    if (outputs) {
      out.push({ title: "outputs", kind: "group", content: Object.entries(outputs).map(([k, v]) => ({
        title: `${k}${v.shape ? ` ${JSON.stringify(v.shape)}` : ""}`, kind: "code", content: String(v.preview ?? ""),
      })) });
    }
    out.push({ title: "attempts", kind: "group", content: attemptSections(ws, rec.attempt_ids ?? []) });
    return out;
  }
  if (nodeId.startsWith("probe_")) {
    const info = rec.result?.probe;
    const out: Section[] = [{ title: "probe", kind: "kv", content: [
      ["status", fmt(rec.status)], ["question", fmt(rec.question)],
      ["source", rec.candidate_id ? `candidate ${rec.candidate_id} (resolved source)` : "new pipeline (writer)"],
      ["out-of-fold score", fmt(info?.score)], ["fold scores", (info?.fold_scores ?? []).map(fmt).join(" / ") || "—"],
      ["rows", fmt(info?.rows)], ["columns", (info?.columns ?? []).join(", ") || "—"],
    ] }];
    if (info?.preview) out.push({ title: "predictions (first rows)", kind: "code", content: info.preview });
    out.push(...graphSection(ws, rec.attempt_ids ?? []));
    out.push({ title: "attempts", kind: "group", content: attemptSections(ws, rec.attempt_ids ?? []) });
    return out;
  }
  if (nodeId.startsWith("setup_")) {
    const out: Section[] = [{ title: "setup", kind: "kv", content: [["status", fmt(rec.status)]] },
      { title: "spec (controller proposal)", kind: "json", content: rec.spec },
      ...graphSection(ws, rec.attempt_ids ?? [])];
    if (rec.result?.outputs) {
      out.push({ title: "audit outputs", kind: "group", content: Object.entries(rec.result.outputs as Record<string, Rec>)
        .map(([k, v]) => ({ title: k, kind: "code", content: String(v.preview ?? "") })) });
    }
    out.push({ title: "attempts", kind: "group", content: attemptSections(ws, rec.attempt_ids ?? []) });
    return out;
  }
  // expansion without candidates
  const out: Section[] = [{ title: "expansion", kind: "kv", content: [
    ["status", fmt(rec.status)], ["parent", fmt(rec.parent_id)], ["error", fmt(rec.error ?? rec.result?.error)],
  ] }];
  if (rec.proposal) out.push({ title: "proposal", kind: "json", content: rec.proposal });
  out.push(...graphSection(ws, rec.attempt_ids ?? []));
  out.push({ title: "attempts", kind: "group", content: attemptSections(ws, rec.attempt_ids ?? []) });
  return out;
}

function findingSections(ws: Workspace, ids: string[], title: string): Section[] {
  if (!ids.length) return [];
  const rows = ids.map((id) => ws.byId.get(id)).filter(Boolean) as Rec[];
  return [{ title: `${title} (${rows.length})`, kind: "text",
    content: rows.map((f) => `• [${f.kind}] ${f.statement}\n  scope: ${f.scope}\n  evidence: ${f.evidence}`).join("\n\n") }];
}

const kTokens = (n: number) => `${(n / 1000).toFixed(1)}k`;

// Calls, tokens and cost of a group of model calls; tokens only where recorded.
function usageText(calls: Rec[]): string {
  const metered = calls.filter((c) => c.usage);
  const sum = (k: string) => metered.reduce((n, c) => n + (c.usage[k] ?? 0), 0);
  const text = [String(calls.length)];
  if (metered.length) {
    text.push(`${kTokens(sum("input_tokens"))} in`,
      `${kTokens(sum("output_tokens"))} out (${kTokens(sum("reasoning_tokens"))} reasoning)`);
    if (metered.every((c) => typeof c.usage.cost === "number")) text.push(`$${sum("cost").toFixed(3)}`);
    if (metered.length < calls.length) text.push(`${calls.length - metered.length} calls unmetered`);
  }
  return text.join(" · ");
}

function overview(ws: Workspace): Section[] {
  const calls = ws.of("model_call");
  const methods = new Map<string, Rec[]>();
  for (const c of calls) methods.set(c.method, [...(methods.get(c.method) ?? []), c]);
  const findings = ws.of("finding");
  const superseded = new Set(findings.flatMap((f) => f.supersedes ?? []));
  return [
    { title: "model calls by method", kind: "kv",
      content: [...[...methods].map(([k, v]): [string, string] => [k, usageText(v)]), ["total", usageText(calls)]] },
    { title: `active findings (${findings.length - superseded.size})`, kind: "text",
      content: findings.filter((f) => !superseded.has(f.id))
        .map((f) => `• [${f.kind}] ${f.statement}  (${f.scope})`).join("\n") },
    { title: "model call transcripts", kind: "files",
      content: listFiles(path.join(ws.dir, "artifacts", "calls"), ws.dir) },
    { title: "workspace files", kind: "files", content: ["report.md", "graph.json", "workspace.json"]
      .filter((f) => existsSync(path.join(ws.dir, f)))
      .map((f) => ({ path: f, size: statSync(path.join(ws.dir, f)).size })) },
  ];
}

// The controller's input is rebuilt from the journal on every call; its transcript
// (artifacts/calls/<id>.input.json) is immutable once written, so sizes are cached.
const CONTEXT_PARTS: [string, string[]][] = [
  ["explorations", ["recent_explorations", "explorations"]],  // window / full memory
  ["leaderboard", ["leaderboard", "solutions", "failed_drafts"]],  // aide: solutions
  ["findings", ["findings"]],
  ["task", ["task"]],
  ["probe outputs", ["probe_outputs"]],
  ["evaluation", ["contract", "locked_evaluation_source"]],
];
const contextCache = new Map<string, number[]>();

// Length of v as Python's json.dumps writes it (", " and ": " separators), which is
// how the harness sends the context to the model.
function pyJsonSize(v: unknown): number {
  if (Array.isArray(v)) return 2 + v.reduce((n: number, x) => n + pyJsonSize(x), 0) + 2 * Math.max(0, v.length - 1);
  if (v && typeof v === "object") {
    const entries = Object.entries(v);
    return 2 + entries.reduce((n, [k, x]) => n + JSON.stringify(k).length + 2 + pyJsonSize(x), 0)
      + 2 * Math.max(0, entries.length - 1);
  }
  return (JSON.stringify(v) ?? "null").length;
}

function contextSizes(ws: Workspace): ContextSizes {
  const started = new Map(ws.events.filter((e) => e.kind === "model_call_started")
    .map((e) => [e.payload.id, e.time]));
  const points = ws.of("model_call").filter((c) => c.method === "control").flatMap((c, i) => {
    const file = path.join(ws.dir, "artifacts", "calls", `${c.id}.input.json`);
    let sizes = contextCache.get(file);
    if (!sizes) {
      const context = readJson(file)?.inputs?.context;
      if (!context || typeof context !== "object") return [];
      const size = (k: string) => (k in context ? pyJsonSize(context[k]) : 0);
      const named = new Set(CONTEXT_PARTS.flatMap(([, keys]) => keys));
      sizes = [...CONTEXT_PARTS.map(([, keys]) => keys.reduce((n, k) => n + size(k), 0)),
        Object.keys(context).filter((k) => !named.has(k)).reduce((n, k) => n + size(k), 0)];
      contextCache.set(file, sizes);
    }
    return [{ order: i + 1, time: started.get(c.id) ?? 0, sizes, tokens: c.usage?.input_tokens }];
  });
  return { parts: [...CONTEXT_PARTS.map(([name]) => name), "other"], points };
}

export const metaMle: Adapter = {
  name: "meta-mle",
  detect(dir) {
    return existsSync(path.join(dir, "state.db")) && existsSync(path.join(dir, "workspace.json"));
  },
  summary(dir, id) {
    return summarize(new Workspace(dir), id);
  },
  detail(dir, id): RunDetail {
    const ws = new Workspace(dir);
    const { nodes, edges } = graph(ws);
    const scores: ScorePoint[] = ws.of("candidate").map((c, i) => ({
      node: c.id, order: i + 1, score: c.status === "ok" ? c.score : null, label: short(c.id),
    }));
    const events: TimelineEvent[] = ws.events
      .filter((e) => e.kind !== "model_call_started")
      .map((e) => ({ time: e.time, kind: e.kind, text: eventText(e.kind, e.payload) }));
    return { summary: summarize(ws, id), nodes, edges, events, scores, overview: overview(ws),
      context: contextSizes(ws) };
  },
  node(dir, nodeId) {
    return nodeSections(new Workspace(dir), nodeId);
  },
};
