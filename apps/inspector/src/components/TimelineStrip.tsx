import type { Timeline, TimelineDay } from "../types";
import { MODE_COLORS, MODE_LABELS, FACILITY_COLORS } from "../lib/colors";

interface Props {
  timeline: Timeline;
  onsetDay: number;
  nDays: number;
}

function cellColor(day: TimelineDay): string {
  if (day.mode === "car" && day.fac) return FACILITY_COLORS[day.fac];
  return MODE_COLORS[day.mode];
}

function Strip({ label, days, onsetDay, nDays }: { label: string; days: TimelineDay[]; onsetDay: number; nDays: number }) {
  const onsetPct = (onsetDay / nDays) * 100;
  return (
    <div className="timeline-block">
      <div className="timeline-arm-label">{label}</div>
      <div className="timeline-strip">
        {days.map((d) => (
          <div
            key={d.d}
            className="timeline-cell"
            style={{ background: cellColor(d) }}
            title={`Day ${d.d}: ${d.mode}${d.fac ? ` via ${d.fac === "T" ? "crossing" : "bypass"}` : ""}`}
          />
        ))}
        <div className="timeline-onset-mark" style={{ left: `${onsetPct}%` }} />
      </div>
    </div>
  );
}

export function TimelineStrip({ timeline, onsetDay, nDays }: Props) {
  return (
    <div>
      <Strip label="Toll arm" days={timeline.toll} onsetDay={onsetDay} nDays={nDays} />
      <Strip label="Placebo arm" days={timeline.placebo} onsetDay={onsetDay} nDays={nDays} />
      <div className="timeline-legend">
        <span className="legend-row">
          <span className="swatch" style={{ background: FACILITY_COLORS.T }} />
          Crossing (car)
        </span>
        <span className="legend-row">
          <span className="swatch" style={{ background: FACILITY_COLORS.R }} />
          Bypass (car)
        </span>
        <span className="legend-row">
          <span className="swatch" style={{ background: MODE_COLORS.transit }} />
          {MODE_LABELS.transit}
        </span>
        <span className="legend-row">
          <span className="swatch" style={{ background: MODE_COLORS.walk }} />
          {MODE_LABELS.walk}
        </span>
        <span className="legend-row">
          <span className="swatch" style={{ background: MODE_COLORS.none }} />
          {MODE_LABELS.none}
        </span>
      </div>
    </div>
  );
}
