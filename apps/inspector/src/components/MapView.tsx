import { useEffect, useRef } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { Agent, CityGeoJSON } from "../types";
import { OUTCOME_COLORS, REWRITE_COLORS, FACILITY_COLORS } from "../lib/colors";
import type { FilterState } from "./FilterBar";

interface Props {
  zones: CityGeoJSON;
  agents: Agent[];
  filters: FilterState;
  colorMode: "outcome" | "rewrite";
  selectedAgentId: string | null;
  onSelectAgent: (id: string) => void;
}

const EMPTY_FC: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };

function carBucket(nCars: number): "0" | "1" | "2+" {
  if (nCars <= 0) return "0";
  if (nCars === 1) return "1";
  return "2+";
}

function agentsToGeoJSON(agents: Agent[]): GeoJSON.FeatureCollection {
  return {
    type: "FeatureCollection",
    features: agents.map((a) => ({
      type: "Feature",
      properties: {
        id: a.id,
        tier: a.tier,
        income_band: a.income_band,
        car_bucket: carBucket(a.n_cars),
        pass_holder: a.pass_holder,
        rewrite_fired: a.rewrite_fired,
        outcome: a.outcome,
      },
      geometry: { type: "Point", coordinates: a.home_xy },
    })),
  };
}

function buildAgentFilter(filters: FilterState): maplibregl.ExpressionSpecification {
  const clauses: maplibregl.ExpressionSpecification[] = [
    ["in", ["get", "tier"], ["literal", Array.from(filters.tiers)]],
    ["in", ["get", "income_band"], ["literal", Array.from(filters.incomeBands)]],
    ["in", ["get", "car_bucket"], ["literal", Array.from(filters.carOwnership)]],
  ];
  if (filters.passHolder !== "any") {
    clauses.push(["==", ["get", "pass_holder"], filters.passHolder === "yes"]);
  }
  if (filters.rewriteFired !== "any") {
    clauses.push(["==", ["get", "rewrite_fired"], filters.rewriteFired === "yes"]);
  }
  return ["all", ...clauses] as unknown as maplibregl.ExpressionSpecification;
}

export function MapView({ zones, agents, filters, colorMode, selectedAgentId, onSelectAgent }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const loadedRef = useRef(false);

  // Init map once.
  useEffect(() => {
    if (!containerRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: {
        version: 8,
        sources: {},
        layers: [
          { id: "bg", type: "background", paint: { "background-color": "#0d0f12" } },
        ],
      },
      center: [0, 0],
      zoom: 12,
      attributionControl: false,
      // No external network access: this style has no glyphs/sprite/tile URLs.
    });
    mapRef.current = map;
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");

    map.on("load", () => {
      loadedRef.current = true;

      map.addSource("zones", { type: "geojson", data: zones });
      map.addSource("agents", { type: "geojson", data: agentsToGeoJSON(agents) });
      map.addSource("selected-line", { type: "geojson", data: EMPTY_FC });

      map.addLayer({
        id: "zones-fill",
        type: "fill",
        source: "zones",
        filter: ["has", "zone_id"],
        paint: { "fill-color": "#1b1f24", "fill-opacity": 0.55 },
      });
      map.addLayer({
        id: "zones-outline",
        type: "line",
        source: "zones",
        filter: ["has", "zone_id"],
        paint: { "line-color": "#2c333b", "line-width": 1 },
      });
      map.addLayer({
        id: "water-fill",
        type: "fill",
        source: "zones",
        filter: ["==", ["get", "kind"], "water"],
        paint: { "fill-color": "#0a2a3a", "fill-opacity": 0.6 },
      });
      map.addLayer({
        id: "bypass-line",
        type: "line",
        source: "zones",
        filter: ["==", ["get", "kind"], "bypass"],
        paint: {
          "line-color": FACILITY_COLORS.R,
          "line-width": 2.5,
          "line-dasharray": [2, 1.5],
        },
      });
      map.addLayer({
        id: "tunnel-line",
        type: "line",
        source: "zones",
        filter: ["==", ["get", "kind"], "tunnel"],
        paint: { "line-color": FACILITY_COLORS.T, "line-width": 3.5 },
      });
      map.addLayer({
        id: "selected-line",
        type: "line",
        source: "selected-line",
        paint: { "line-color": "#e8eaed", "line-width": 1, "line-dasharray": [1, 1] },
      });
      map.addLayer({
        id: "agent-dots",
        type: "circle",
        source: "agents",
        paint: {
          "circle-radius": ["case", ["==", ["get", "id"], selectedAgentId ?? "__none__"], 6, 3.4],
          "circle-color": outcomeColorExpr(),
          "circle-stroke-color": "#0d0f12",
          "circle-stroke-width": 1,
        },
      });
      map.setFilter("agent-dots", buildAgentFilter(filters));

      map.on("click", "agent-dots", (e) => {
        const f = e.features?.[0];
        const id = f?.properties?.id;
        if (id) onSelectAgent(id as string);
      });
      map.on("mouseenter", "agent-dots", () => {
        map.getCanvas().style.cursor = "pointer";
      });
      map.on("mouseleave", "agent-dots", () => {
        map.getCanvas().style.cursor = "";
      });

      // Fit to zone bounds.
      const bounds = new maplibregl.LngLatBounds();
      for (const f of zones.features) {
        if (f.geometry.type === "Polygon") {
          for (const ring of f.geometry.coordinates) {
            for (const c of ring) bounds.extend(c as [number, number]);
          }
        } else if (f.geometry.type === "LineString") {
          for (const c of f.geometry.coordinates) bounds.extend(c as [number, number]);
        }
      }
      map.fitBounds(bounds, { padding: 40, duration: 0 });
    });

    return () => {
      map.remove();
      mapRef.current = null;
      loadedRef.current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update agent color paint on colorMode change.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !loadedRef.current) return;
    const expr = colorMode === "outcome" ? outcomeColorExpr() : rewriteColorExpr();
    if (map.getLayer("agent-dots")) {
      map.setPaintProperty("agent-dots", "circle-color", expr);
    }
  }, [colorMode]);

  // Update filter.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !loadedRef.current) return;
    if (map.getLayer("agent-dots")) {
      map.setFilter("agent-dots", buildAgentFilter(filters));
    }
  }, [filters]);

  // Update selection highlight + home-work line.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !loadedRef.current) return;
    if (map.getLayer("agent-dots")) {
      map.setPaintProperty("agent-dots", "circle-radius", [
        "case",
        ["==", ["get", "id"], selectedAgentId ?? "__none__"],
        6,
        3.4,
      ]);
    }
    const src = map.getSource("selected-line") as maplibregl.GeoJSONSource | undefined;
    if (src) {
      const agent = agents.find((a) => a.id === selectedAgentId);
      if (agent) {
        src.setData({
          type: "FeatureCollection",
          features: [
            {
              type: "Feature",
              properties: {},
              geometry: {
                type: "LineString",
                coordinates: [agent.home_xy, agent.work_xy],
              },
            },
          ],
        });
      } else {
        src.setData(EMPTY_FC);
      }
    }
  }, [selectedAgentId, agents]);

  return (
    <div className="map-pane">
      <div ref={containerRef} className="maplibre-map" />
      <div className="map-legend">
        {colorMode === "outcome" ? (
          <>
            {Object.entries(OUTCOME_COLORS).map(([k, color]) => (
              <div className="legend-row" key={k}>
                <span className="swatch" style={{ background: color }} />
                {k.replace("_", " ")}
              </div>
            ))}
          </>
        ) : (
          <>
            <div className="legend-row">
              <span className="swatch" style={{ background: REWRITE_COLORS.fired }} />
              rewrite fired
            </div>
            <div className="legend-row">
              <span className="swatch" style={{ background: REWRITE_COLORS.not_fired }} />
              no rewrite
            </div>
          </>
        )}
        <div className="legend-row" style={{ marginTop: 4 }}>
          <span className="line-swatch" style={{ background: FACILITY_COLORS.T }} />
          crossing corridor
        </div>
        <div className="legend-row">
          <span className="line-swatch" style={{ background: FACILITY_COLORS.R }} />
          bypass route
        </div>
      </div>
    </div>
  );
}

function outcomeColorExpr(): maplibregl.ExpressionSpecification {
  return [
    "match",
    ["get", "outcome"],
    "kept_tunnel",
    OUTCOME_COLORS.kept_tunnel,
    "diverted",
    OUTCOME_COLORS.diverted,
    "mode_change",
    OUTCOME_COLORS.mode_change,
    "suppressed",
    OUTCOME_COLORS.suppressed,
    "#888888",
  ] as unknown as maplibregl.ExpressionSpecification;
}

function rewriteColorExpr(): maplibregl.ExpressionSpecification {
  return [
    "case",
    ["==", ["get", "rewrite_fired"], true],
    REWRITE_COLORS.fired,
    REWRITE_COLORS.not_fired,
  ] as unknown as maplibregl.ExpressionSpecification;
}
