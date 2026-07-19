import { useMemo, useState } from "react";
import "./App.css";
import { useCityData } from "./lib/useCityData";
import { FilterBar, emptyFilterState, type FilterState } from "./components/FilterBar";
import { MapView } from "./components/MapView";
import { AgentDrawer } from "./components/AgentDrawer";
import { AggregatePanel } from "./components/AggregatePanel";
import type { Agent } from "./types";

function carBucket(nCars: number): "0" | "1" | "2+" {
  if (nCars <= 0) return "0";
  if (nCars === 1) return "1";
  return "2+";
}

function App() {
  const { data, error, loading } = useCityData();
  const [filters, setFilters] = useState<FilterState | null>(null);
  const [colorMode, setColorMode] = useState<"outcome" | "rewrite">("outcome");
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);

  const effectiveFilters = useMemo(() => {
    if (filters) return filters;
    if (!data) return null;
    return emptyFilterState(data.meta.tiers, data.meta.income_bands);
  }, [filters, data]);

  const filteredAgents = useMemo<Agent[]>(() => {
    if (!data || !effectiveFilters) return [];
    return data.agents.filter((a) => {
      if (!effectiveFilters.tiers.has(a.tier)) return false;
      if (!effectiveFilters.incomeBands.has(a.income_band)) return false;
      if (!effectiveFilters.carOwnership.has(carBucket(a.n_cars))) return false;
      if (effectiveFilters.passHolder !== "any") {
        const want = effectiveFilters.passHolder === "yes";
        if (a.pass_holder !== want) return false;
      }
      if (effectiveFilters.rewriteFired !== "any") {
        const want = effectiveFilters.rewriteFired === "yes";
        if (a.rewrite_fired !== want) return false;
      }
      return true;
    });
  }, [data, effectiveFilters]);

  const selectedAgent = useMemo(
    () => data?.agents.find((a) => a.id === selectedAgentId) ?? null,
    [data, selectedAgentId]
  );

  if (loading) {
    return (
      <div className="app-shell">
        <div className="app-header">
          <h1>City K — Agent Inspector</h1>
        </div>
        <div style={{ padding: 24, color: "var(--text-1)" }}>Loading sample data…</div>
      </div>
    );
  }

  if (error || !data || !effectiveFilters) {
    return (
      <div className="app-shell">
        <div className="app-header">
          <h1>City K — Agent Inspector</h1>
        </div>
        <div style={{ padding: 24, color: "var(--bad)" }}>
          Failed to load data from public/data/: {error ?? "unknown error"}
        </div>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <div className="app-header">
        <div>
          <h1>City K — Agent Inspector</h1>
          <span className="subtitle">
            view-only · masked simulation · onset day {data.meta.onset_day} of {data.meta.n_days}
          </span>
        </div>
        <span className="status">
          {data.agents.length} agents loaded (sample data — see README.md)
        </span>
      </div>

      <FilterBar
        tiers={data.meta.tiers}
        incomeBands={data.meta.income_bands}
        filters={effectiveFilters}
        onChange={setFilters}
        colorMode={colorMode}
        onColorModeChange={setColorMode}
      />

      <div className="main-area">
        <MapView
          zones={data.zones}
          agents={data.agents}
          filters={effectiveFilters}
          colorMode={colorMode}
          selectedAgentId={selectedAgentId}
          onSelectAgent={setSelectedAgentId}
        />
        <AgentDrawer agent={selectedAgent} meta={data.meta} onClose={() => setSelectedAgentId(null)} />
      </div>

      <AggregatePanel
        aggregate={data.aggregate}
        meta={data.meta}
        filteredCount={filteredAgents.length}
        totalCount={data.agents.length}
      />
    </div>
  );
}

export default App;
