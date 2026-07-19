import { useEffect, useState } from "react";
import type { Aggregate, Agent, CityGeoJSON, Meta } from "../types";

export interface CityData {
  meta: Meta;
  zones: CityGeoJSON;
  agents: Agent[];
  aggregate: Aggregate;
}

interface State {
  data: CityData | null;
  error: string | null;
  loading: boolean;
}

const BASE = import.meta.env.BASE_URL;

/** Loads the four data-contract files from public/data/. Read-only, local. */
export function useCityData(): State {
  const [state, setState] = useState<State>({ data: null, error: null, loading: true });

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [meta, zones, agents, aggregate] = await Promise.all([
          fetch(`${BASE}data/meta.json`).then((r) => r.json()),
          fetch(`${BASE}data/zones.geojson`).then((r) => r.json()),
          fetch(`${BASE}data/agents.json`).then((r) => r.json()),
          fetch(`${BASE}data/aggregate.json`).then((r) => r.json()),
        ]);
        if (!cancelled) {
          setState({ data: { meta, zones, agents, aggregate }, error: null, loading: false });
        }
      } catch (err) {
        if (!cancelled) {
          setState({
            data: null,
            error: err instanceof Error ? err.message : String(err),
            loading: false,
          });
        }
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  return state;
}
