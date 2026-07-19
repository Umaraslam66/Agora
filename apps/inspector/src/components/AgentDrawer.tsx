import type { Agent, Meta } from "../types";
import { OUTCOME_COLORS, OUTCOME_LABELS } from "../lib/colors";
import { PersonaDiff } from "./PersonaDiff";
import { HabitChart } from "./HabitChart";
import { TimelineStrip } from "./TimelineStrip";

interface Props {
  agent: Agent | null;
  meta: Meta;
  onClose: () => void;
}

export function AgentDrawer({ agent, meta, onClose }: Props) {
  if (!agent) {
    return (
      <aside className="drawer">
        <p className="empty-drawer">Click an agent dot on the map to inspect its persona card,
        habit strengths, and BT1-window choice timeline.</p>
      </aside>
    );
  }

  return (
    <aside className="drawer">
      <div className="drawer-header">
        <h2>{agent.id}</h2>
        <button className="drawer-close" type="button" onClick={onClose} aria-label="Close">
          ×
        </button>
      </div>

      <section>
        <h3>Summary</h3>
        <div className="agent-meta-grid">
          <span className="k">Tier</span>
          <span className="v">{agent.tier}</span>
          <span className="k">Home / work zone</span>
          <span className="v">
            {agent.home_zone} → {agent.work_zone}
          </span>
          <span className="k">Income band</span>
          <span className="v">{agent.income_band}</span>
          <span className="k">Cars</span>
          <span className="v">{agent.n_cars}</span>
          <span className="k">Pass holder</span>
          <span className="v">{agent.pass_holder ? "Yes" : "No"}</span>
          <span className="k">Corridor traveler</span>
          <span className="v">{agent.corridor_traveler ? "Yes" : "No"}</span>
          <span className="k">Rewrite fired</span>
          <span className="v">{agent.rewrite_fired ? "Yes" : "No"}</span>
          <span className="k">Outcome</span>
          <span className="v">
            <span
              className="outcome-badge"
              style={{ background: OUTCOME_COLORS[agent.outcome] }}
            >
              {OUTCOME_LABELS[agent.outcome]}
            </span>
          </span>
        </div>
      </section>

      <section>
        <h3>Persona card — before → after onset</h3>
        <PersonaDiff before={agent.card_before} after={agent.card_after} />
      </section>

      <section>
        <h3>Habit strength by rule</h3>
        <HabitChart habit={agent.habit} onsetDay={meta.onset_day} nDays={meta.n_days} />
      </section>

      <section>
        <h3>Daily choice timeline (BT1 window)</h3>
        <TimelineStrip timeline={agent.timeline} onsetDay={meta.onset_day} nDays={meta.n_days} />
      </section>
    </aside>
  );
}
