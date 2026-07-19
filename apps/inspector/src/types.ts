// Data-contract types for the City K agent inspector.
// These mirror the shapes documented in README.md exactly; the real export
// will conform to the same shapes, so no other file should redefine them.

export type Tier = "T1" | "T2" | "T3" | "T4" | "T4_noclaims" | "T5";
export type IncomeBand = string;
export type Outcome = "kept_tunnel" | "diverted" | "mode_change" | "suppressed";
export type Mode = "car" | "transit" | "walk" | "none";
export type Facility = "T" | "R" | null;

export interface Meta {
  onset_day: number;
  n_days: number;
  warmup_days: number;
  tiers: Tier[];
  income_bands: IncomeBand[];
  outcomes: Outcome[];
}

export interface ZoneProperties {
  zone_id: string;
}

export interface RouteProperties {
  kind: "tunnel" | "bypass" | "water";
}

export type CityFeature = GeoJSON.Feature<
  GeoJSON.Polygon | GeoJSON.LineString,
  ZoneProperties | RouteProperties
>;

export interface CityGeoJSON extends GeoJSON.FeatureCollection {
  features: CityFeature[];
}

export interface HabitSeries {
  rule: string;
  series: [day: number, strength: number][];
}

export interface TimelineDay {
  d: number;
  mode: Mode;
  fac: Facility;
}

export interface Timeline {
  toll: TimelineDay[];
  placebo: TimelineDay[];
}

export interface Agent {
  id: string;
  tier: Tier;
  home_zone: string;
  work_zone: string;
  home_xy: [number, number];
  work_xy: [number, number];
  income_band: IncomeBand;
  n_cars: number;
  pass_holder: boolean;
  corridor_traveler: boolean;
  rewrite_fired: boolean;
  outcome: Outcome;
  card_before: string[];
  card_after: string[] | null;
  habit: HabitSeries[];
  timeline: Timeline;
}

export interface Aggregate {
  days: number[];
  tunnel: {
    toll: number[];
    placebo: number[];
  };
}
