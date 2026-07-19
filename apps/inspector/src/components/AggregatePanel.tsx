import type { Aggregate, Meta } from "../types";

interface Props {
  aggregate: Aggregate;
  meta: Meta;
  filteredCount: number;
  totalCount: number;
}

const W = 640;
const H = 130;
const PAD_L = 34;
const PAD_R = 10;
const PAD_T = 8;
const PAD_B = 18;

function buildPath(values: number[], days: number[], maxV: number): string {
  const innerW = W - PAD_L - PAD_R;
  const innerH = H - PAD_T - PAD_B;
  return values
    .map((v, i) => {
      const x = PAD_L + (days[i] / Math.max(1, days.length - 1)) * innerW;
      const y = PAD_T + innerH - (v / maxV) * innerH;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

export function AggregatePanel({ aggregate, meta, filteredCount, totalCount }: Props) {
  const { days, tunnel } = aggregate;
  const maxV = Math.max(...tunnel.toll, ...tunnel.placebo) * 1.08;
  const onsetX = PAD_L + (meta.onset_day / Math.max(1, days.length - 1)) * (W - PAD_L - PAD_R);

  return (
    <div className="aggregate-panel">
      <div className="agg-meta">
        <div>
          <div className="agg-title">Agents matching filters</div>
          <div className="agg-count">
            {filteredCount} / {totalCount}
          </div>
        </div>
        <div className="agg-note">
          Crossing volume below reflects all agents, unaffected by filters — labeled accordingly.
        </div>
      </div>
      <div className="agg-chart-wrap">
        <div className="agg-title" style={{ marginBottom: 4 }}>
          Daily crossing-corridor volume — all agents
        </div>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="none">
          {/* axis */}
          <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={H - PAD_B} stroke="var(--border)" />
          <line
            x1={PAD_L}
            y1={H - PAD_B}
            x2={W - PAD_R}
            y2={H - PAD_B}
            stroke="var(--border)"
          />
          {/* onset marker */}
          <line
            x1={onsetX}
            x2={onsetX}
            y1={PAD_T}
            y2={H - PAD_B}
            stroke="#ffffff"
            strokeOpacity={0.4}
            strokeDasharray="3,3"
          />
          <text x={onsetX + 3} y={PAD_T + 9} fill="var(--text-2)" fontSize="9">
            onset (day {meta.onset_day})
          </text>
          {/* series */}
          <path d={buildPath(tunnel.placebo, days, maxV)} fill="none" stroke="#aab0b8" strokeWidth={1.4} />
          <path d={buildPath(tunnel.toll, days, maxV)} fill="none" stroke="#56b4e9" strokeWidth={1.8} />
        </svg>
        <div className="agg-legend">
          <span className="legend-row">
            <span className="line-swatch" style={{ background: "#56b4e9" }} />
            Toll arm
          </span>
          <span className="legend-row">
            <span className="line-swatch" style={{ background: "#aab0b8" }} />
            Placebo arm
          </span>
        </div>
      </div>
    </div>
  );
}
