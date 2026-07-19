import type { IncomeBand, Tier } from "../types";

export type CarOwnershipKey = "0" | "1" | "2+";

export interface FilterState {
  tiers: Set<Tier>;
  incomeBands: Set<IncomeBand>;
  carOwnership: Set<CarOwnershipKey>;
  passHolder: "any" | "yes" | "no";
  rewriteFired: "any" | "yes" | "no";
}

export function emptyFilterState(tiers: Tier[], incomeBands: IncomeBand[]): FilterState {
  return {
    tiers: new Set(tiers),
    incomeBands: new Set(incomeBands),
    carOwnership: new Set(["0", "1", "2+"]),
    passHolder: "any",
    rewriteFired: "any",
  };
}

interface Props {
  tiers: Tier[];
  incomeBands: IncomeBand[];
  filters: FilterState;
  onChange: (next: FilterState) => void;
  colorMode: "outcome" | "rewrite";
  onColorModeChange: (mode: "outcome" | "rewrite") => void;
}

function toggleInSet<T>(set: Set<T>, value: T): Set<T> {
  const next = new Set(set);
  if (next.has(value)) next.delete(value);
  else next.add(value);
  return next;
}

export function FilterBar({
  tiers,
  incomeBands,
  filters,
  onChange,
  colorMode,
  onColorModeChange,
}: Props) {
  const carKeys: CarOwnershipKey[] = ["0", "1", "2+"];

  return (
    <div className="filter-bar">
      <div className="filter-group">
        <span className="filter-label">Tier</span>
        <div className="pill-row">
          {tiers.map((t) => (
            <button
              key={t}
              className={`pill ${filters.tiers.has(t) ? "active" : ""}`}
              onClick={() => onChange({ ...filters, tiers: toggleInSet(filters.tiers, t) })}
              type="button"
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      <div className="filter-group">
        <span className="filter-label">Income band</span>
        <div className="pill-row">
          {incomeBands.map((b) => (
            <button
              key={b}
              className={`pill ${filters.incomeBands.has(b) ? "active" : ""}`}
              onClick={() =>
                onChange({ ...filters, incomeBands: toggleInSet(filters.incomeBands, b) })
              }
              type="button"
            >
              {b}
            </button>
          ))}
        </div>
      </div>

      <div className="filter-group">
        <span className="filter-label">Car ownership</span>
        <div className="pill-row">
          {carKeys.map((c) => (
            <button
              key={c}
              className={`pill ${filters.carOwnership.has(c) ? "active" : ""}`}
              onClick={() =>
                onChange({ ...filters, carOwnership: toggleInSet(filters.carOwnership, c) })
              }
              type="button"
            >
              {c}
            </button>
          ))}
        </div>
      </div>

      <div className="filter-group">
        <label className="filter-label" htmlFor="filter-pass-holder">
          Pass holder
        </label>
        <select
          id="filter-pass-holder"
          name="pass-holder"
          value={filters.passHolder}
          onChange={(e) =>
            onChange({ ...filters, passHolder: e.target.value as FilterState["passHolder"] })
          }
        >
          <option value="any">Any</option>
          <option value="yes">Yes</option>
          <option value="no">No</option>
        </select>
      </div>

      <div className="filter-group">
        <label className="filter-label" htmlFor="filter-rewrite-fired">
          Rewrite fired
        </label>
        <select
          id="filter-rewrite-fired"
          name="rewrite-fired"
          value={filters.rewriteFired}
          onChange={(e) =>
            onChange({ ...filters, rewriteFired: e.target.value as FilterState["rewriteFired"] })
          }
        >
          <option value="any">Any</option>
          <option value="yes">Yes</option>
          <option value="no">No</option>
        </select>
      </div>

      <button
        type="button"
        className="reset-btn"
        onClick={() => onChange(emptyFilterState(tiers, incomeBands))}
      >
        Reset filters
      </button>

      <div className="filter-group color-mode-toggle">
        <span className="filter-label">Color dots by</span>
        <div className="pill-row">
          <button
            type="button"
            className={`pill ${colorMode === "outcome" ? "active" : ""}`}
            onClick={() => onColorModeChange("outcome")}
          >
            Outcome
          </button>
          <button
            type="button"
            className={`pill ${colorMode === "rewrite" ? "active" : ""}`}
            onClick={() => onColorModeChange("rewrite")}
          >
            Rewrite fired
          </button>
        </div>
      </div>
    </div>
  );
}
