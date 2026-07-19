import type { HabitSeries } from "../types";

interface Props {
  habit: HabitSeries[];
  onsetDay: number;
  nDays: number;
}

const W = 340;
const H = 44;
const PAD = 4;
const MAX_STRENGTH = 14; // matches the provisional STRONG_HABIT_THRESHOLD scale

function buildPath(series: [number, number][], nDays: number): string {
  if (series.length === 0) return "";
  return series
    .map(([day, strength], i) => {
      const x = PAD + (day / Math.max(1, nDays - 1)) * (W - 2 * PAD);
      const y = H - PAD - (Math.min(strength, MAX_STRENGTH) / MAX_STRENGTH) * (H - 2 * PAD);
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

export function HabitChart({ habit, onsetDay, nDays }: Props) {
  if (habit.length === 0) {
    return <p className="no-rewrite-note">No habit-strength series recorded for this agent.</p>;
  }
  const onsetX = PAD + (onsetDay / Math.max(1, nDays - 1)) * (W - 2 * PAD);

  return (
    <>
      {habit.map((h, idx) => (
        <div className="habit-rule" key={idx}>
          <div className="habit-rule-label">{h.rule}</div>
          <svg
            className="habit-chart"
            viewBox={`0 0 ${W} ${H}`}
            preserveAspectRatio="none"
            role="img"
            aria-label={`Habit strength over time for rule: ${h.rule}`}
          >
            <line
              x1={onsetX}
              x2={onsetX}
              y1={0}
              y2={H}
              stroke="#ffffff"
              strokeOpacity={0.35}
              strokeWidth={1}
              strokeDasharray="2,2"
            />
            <path d={buildPath(h.series, nDays)} fill="none" stroke="#56b4e9" strokeWidth={1.5} />
          </svg>
        </div>
      ))}
    </>
  );
}
