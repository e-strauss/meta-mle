"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { score as fmtScore } from "@/lib/format";
import { layout, type Placed, X_STEP, Y_STEP } from "@/lib/tree-layout";
import type { EdgeKind, GraphEdge, GraphNode } from "@/lib/types";

const R = 8;
const RAMP = ["var(--seq-0)", "var(--seq-1)", "var(--seq-2)", "var(--seq-3)", "var(--seq-4)"];

const EDGE_STYLE: Record<EdgeKind, { dash?: string; width: number; color: string; label: string }> = {
  primary: { width: 1.5, color: "var(--edge)", label: "parent → child" },
  requested: { dash: "2 3", width: 1, color: "var(--edge)", label: "requested exploration / setup" },
  evidence: { dash: "6 4", width: 1, color: "var(--series-2)", label: "evidence used" },
  reference: { dash: "6 4", width: 1.5, color: "var(--series-1)", label: "reference" },
};

function fill(n: GraphNode, lo: number, hi: number): string {
  if (n.status === "failed" || n.status === "rejected") return "var(--critical)";
  if (n.status === "running") return "var(--warning)";
  if (n.status === "interrupted") return "var(--serious)";
  if (typeof n.score === "number") {
    const t = hi > lo ? (n.score - lo) / (hi - lo) : 1;
    return RAMP[Math.min(RAMP.length - 1, Math.floor(t * RAMP.length))];
  }
  return "var(--neutral-node)";
}

function Shape({ n, color, selected }: { n: Placed; color: string; selected: boolean }) {
  const ring = selected ? "var(--fg)" : "var(--chart-surface)";
  const sw = selected ? 2.5 : 2;
  const cls = n.status === "running" ? "pulse" : undefined;
  if (n.shape === "diamond") {
    const d = R + 1;
    return <path className={cls} d={`M${n.x} ${n.y - d}L${n.x + d} ${n.y}L${n.x} ${n.y + d}L${n.x - d} ${n.y}Z`}
      fill={color} stroke={ring} strokeWidth={sw} />;
  }
  if (n.shape === "square") {
    return <rect className={cls} x={n.x - R} y={n.y - R} width={2 * R} height={2 * R} rx={3} fill={color} stroke={ring} strokeWidth={sw} />;
  }
  return <circle className={cls} cx={n.x} cy={n.y} r={R} fill={color} stroke={ring} strokeWidth={sw} />;
}

function glyph(n: GraphNode): string {
  if (n.status === "failed" || n.status === "rejected") return "✕ ";
  if (n.status === "running") return "⋯ ";
  if (n.status === "interrupted") return "‖ ";
  return "";
}

function edgePath(a: Placed, b: Placed): string {
  if (a.x === b.x) {
    const bend = a.x + X_STEP * 0.3;
    return `M${a.x + R} ${a.y}C${bend} ${a.y} ${bend} ${b.y} ${b.x + R} ${b.y}`;
  }
  const mid = (a.x + b.x) / 2;
  return `M${a.x + R} ${a.y}C${mid} ${a.y} ${mid} ${b.y} ${b.x - R} ${b.y}`;
}

export default function TreeView({ nodes, edges, selected, onSelect, bestId }: {
  nodes: GraphNode[]; edges: GraphEdge[]; selected: string | null; onSelect: (id: string) => void; bestId?: string;
}) {
  const { placed, width, height } = useMemo(() => layout(nodes), [nodes]);
  const scores = nodes.map((n) => n.score).filter((s): s is number => typeof s === "number");
  const lo = Math.min(...scores), hi = Math.max(...scores);
  const pad = 40;
  const left = 150; // room for centred labels of the root column
  const full = { x: -left, y: -pad, w: width + X_STEP * 0.6 + left, h: height + 2 * pad };
  const [view, setView] = useState<typeof full | null>(null);
  const v = view ?? full;
  const treeRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const drag = useRef<{ x: number; y: number; v: typeof full } | null>(null);
  const [hover, setHover] = useState<{ n: Placed; px: number; py: number } | null>(null);
  const [hidden, setHidden] = useState<Set<EdgeKind>>(new Set());
  const [fullscreen, setFullscreen] = useState(false);
  const [fullscreenAvailable, setFullscreenAvailable] = useState(false);
  const [fullscreenError, setFullscreenError] = useState<string | null>(null);
  const inner = useMemo(() => new Set(nodes.map((n) => n.parent).filter(Boolean) as string[]), [nodes]);
  const best = bestId;

  useEffect(() => {
    setFullscreenAvailable(Boolean(document.fullscreenEnabled && treeRef.current?.requestFullscreen));
    const onFullscreenChange = () => {
      setFullscreen(document.fullscreenElement === treeRef.current);
      setHover(null);
      drag.current = null;
    };
    document.addEventListener("fullscreenchange", onFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", onFullscreenChange);
  }, []);

  const toggleFullscreen = async () => {
    setFullscreenError(null);
    try {
      if (document.fullscreenElement === treeRef.current) {
        await document.exitFullscreen();
      } else {
        await treeRef.current?.requestFullscreen();
      }
    } catch {
      setFullscreenError("Could not change fullscreen mode. Please try again.");
    }
  };

  const toUnits = (dx: number, dy: number) => {
    const rect = svgRef.current!.getBoundingClientRect();
    const s = Math.max(v.w / rect.width, v.h / rect.height);
    return [dx * s, dy * s];
  };
  // Plain wheel / two-finger scroll scrolls the page. Zoom only on pinch (reported
  // as ctrlKey wheel events) or Ctrl/⌘ + wheel. Native non-passive listener, so
  // the pinch can be kept from zooming the whole page.
  const zoomRef = useRef<(factor: number, cx: number, cy: number) => void>(() => {});
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const onWheel = (e: WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      const rect = svg.getBoundingClientRect();
      zoomRef.current(e.deltaY > 0 ? 1.12 : 0.89, (e.clientX - rect.left) / rect.width, (e.clientY - rect.top) / rect.height);
    };
    svg.addEventListener("wheel", onWheel, { passive: false });
    return () => svg.removeEventListener("wheel", onWheel);
  }, []);

  const zoom = (factor: number, cx = 0.5, cy = 0.5) => {
    const w = v.w * factor, h = v.h * factor;
    setView({ x: v.x + (v.w - w) * cx, y: v.y + (v.h - h) * cy, w, h });
  };
  zoomRef.current = zoom;

  const groups = new Map<string, Placed[]>();
  for (const n of placed.values()) if (n.group) {
    if (!groups.has(n.group)) groups.set(n.group, []);
    groups.get(n.group)!.push(n);
  }

  return (
    <>
      <div className="legend">
        <span><svg width="64" height="12">{RAMP.map((c, i) => <rect key={i} x={i * 13} y={1} width={12} height={10} rx={2} fill={c} />)}</svg> score low → high</span>
        <span><svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="var(--critical)" /></svg> ✕ failed / rejected</span>
        <span><svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="var(--warning)" /></svg> ⋯ running</span>
        <span><svg width="16" height="16"><circle cx="8" cy="8" r="3" fill="var(--seq-3)" /><circle cx="8" cy="8" r="6.5" fill="none" stroke="var(--good)" strokeWidth="1.5" /></svg> best</span>
        <span>◆ exploration · ■ root / setup · ● candidate</span>
        {(Object.keys(EDGE_STYLE) as EdgeKind[]).filter((k) => edges.some((e) => e.kind === k)).map((k) => (
          <label key={k} style={{ cursor: "pointer", opacity: hidden.has(k) ? 0.45 : 1 }}>
            <input type="checkbox" checked={!hidden.has(k)} onChange={() => {
              const next = new Set(hidden);
              if (next.has(k)) next.delete(k); else next.add(k);
              setHidden(next);
            }} style={{ verticalAlign: "middle" }} />{" "}
            <svg width="26" height="8"><line x1="0" y1="4" x2="26" y2="4" stroke={EDGE_STYLE[k].color}
              strokeWidth={EDGE_STYLE[k].width + 0.5} strokeDasharray={EDGE_STYLE[k].dash} /></svg> {EDGE_STYLE[k].label}
          </label>
        ))}
      </div>
      <div ref={treeRef} className="tree-wrap">
        <div className="tree-controls">
          <button onClick={() => zoom(0.8)} aria-label="Zoom in">+</button>
          <button onClick={() => zoom(1.25)} aria-label="Zoom out">−</button>
          <button onClick={() => setView(null)}>Fit</button>
          <button onClick={toggleFullscreen} disabled={!fullscreenAvailable}
            aria-label={fullscreen ? "Exit fullscreen" : "Enter fullscreen"} aria-pressed={fullscreen}
            title={fullscreenAvailable ? (fullscreen ? "Exit fullscreen (Esc)" : "Show search tree fullscreen") : "Fullscreen is unavailable in this browser"}>
            {fullscreen ? "Exit fullscreen" : "Fullscreen"}
          </button>
          <span className="small muted" style={{ alignSelf: "center" }}>pinch or ⌘/Ctrl + scroll to zoom · drag to pan</span>
        </div>
        <svg ref={svgRef} viewBox={`${v.x} ${v.y} ${v.w} ${v.h}`} role="img" aria-label="Search tree"
          onPointerDown={(e) => { drag.current = { x: e.clientX, y: e.clientY, v }; }}
          onPointerMove={(e) => {
            if (!drag.current) return;
            const [dx, dy] = toUnits(e.clientX - drag.current.x, e.clientY - drag.current.y);
            setView({ ...drag.current.v, x: drag.current.v.x - dx, y: drag.current.v.y - dy });
          }}
          onPointerUp={() => { drag.current = null; }}
          onPointerLeave={() => { drag.current = null; }}>
          {[...groups.entries()].map(([g, members]) => {
            const ys = members.map((m) => m.y);
            const x = members[0].x;
            return <rect key={g} x={x - R - 6} y={Math.min(...ys) - R - 6} width={X_STEP - 30}
              height={Math.max(...ys) - Math.min(...ys) + 2 * R + 12} rx={8} fill="none"
              stroke="var(--grid)" strokeWidth={1.5} />;
          })}
          {edges.filter((e) => !hidden.has(e.kind)).map((e, i) => {
            const a = placed.get(e.from), b = placed.get(e.to);
            if (!a || !b) return null;
            const s = EDGE_STYLE[e.kind];
            return <path key={i} d={edgePath(a, b)} fill="none" stroke={s.color} strokeWidth={s.width} strokeDasharray={s.dash} />;
          })}
          {[...placed.values()].map((n) => (
            <g key={n.id} onClick={() => onSelect(n.id)} style={{ cursor: "pointer" }}
              onPointerEnter={(e) => {
                const rect = svgRef.current!.getBoundingClientRect();
                setHover({ n, px: e.clientX - rect.left, py: e.clientY - rect.top });
              }}
              onPointerLeave={() => setHover(null)}>
              <rect x={n.x - R - 4} y={n.y - Y_STEP / 2} width={X_STEP - 40} height={Y_STEP} fill="transparent" />
              <Shape n={n} color={fill(n, lo, hi)} selected={selected === n.id} />
              {best === n.id && <circle cx={n.x} cy={n.y} r={R + 5} fill="none" stroke="var(--good)" strokeWidth={2} />}
              {inner.has(n.id) ? (
                // Inner nodes: label above, subtitle below, clear of outgoing edges.
                <>
                  <text x={n.x} y={n.y - R - 7} textAnchor="middle">
                    {glyph(n)}{n.label}{typeof n.score === "number" ? `  ${fmtScore(n.score)}` : ""}
                  </text>
                  {n.subtitle && <text className="sub" x={n.x} y={n.y + R + 15} textAnchor="middle">
                    {n.subtitle.length > 34 ? n.subtitle.slice(0, 33) + "…" : n.subtitle}
                  </text>}
                </>
              ) : (
                <>
                  <text x={n.x + R + 6} y={n.y - 2}>
                    {glyph(n)}{n.label}{typeof n.score === "number" ? `  ${fmtScore(n.score)}` : ""}
                  </text>
                  {n.subtitle && <text className="sub" x={n.x + R + 6} y={n.y + 12}>
                    {n.subtitle.length > 34 ? n.subtitle.slice(0, 33) + "…" : n.subtitle}
                  </text>}
                </>
              )}
            </g>
          ))}
        </svg>
        {fullscreenError && <div className="tree-error error small" role="alert">{fullscreenError}</div>}
        {hover && (
          <div className="tooltip" style={{ left: Math.min(hover.px + 14, 9999), top: hover.py + 14 }}>
            <div><b>{hover.n.label}</b> <span className="muted">{hover.n.kind} · {hover.n.status}</span></div>
            {typeof hover.n.score === "number" && <div>score {fmtScore(hover.n.score)}</div>}
            {hover.n.subtitle && <div className="muted">{hover.n.subtitle}</div>}
            <div className="muted small">click for details</div>
          </div>
        )}
      </div>
    </>
  );
}
