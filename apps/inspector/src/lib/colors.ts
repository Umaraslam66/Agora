// Colorblind-safe palette (Okabe-Ito) used consistently across the map,
// timeline strips, and legends.
import type { Mode, Outcome } from "../types";

export const OUTCOME_COLORS: Record<Outcome, string> = {
  kept_tunnel: "#0072B2", // blue
  diverted: "#E69F00", // orange
  mode_change: "#009E73", // bluish green
  suppressed: "#D55E00", // vermillion
};

export const OUTCOME_LABELS: Record<Outcome, string> = {
  kept_tunnel: "Kept crossing",
  diverted: "Diverted to bypass",
  mode_change: "Changed mode",
  suppressed: "Suppressed trip",
};

export const REWRITE_COLORS: Record<"fired" | "not_fired", string> = {
  fired: "#F0E442", // yellow
  not_fired: "#555555",
};

export const MODE_COLORS: Record<Mode, string> = {
  car: "#0072B2",
  transit: "#009E73",
  walk: "#E69F00",
  none: "#3a3a3a",
};

export const MODE_LABELS: Record<Mode, string> = {
  car: "Car",
  transit: "Transit",
  walk: "Walk",
  none: "No trip",
};

export const FACILITY_COLORS: Record<"T" | "R", string> = {
  T: "#D55E00", // tunnel — vermillion
  R: "#56B4E9", // bypass — sky blue
};
