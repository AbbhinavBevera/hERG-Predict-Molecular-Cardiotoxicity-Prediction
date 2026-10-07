import React from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  ReferenceLine,
} from "recharts";
import type { FragmentAttribution } from "../types";

interface Props {
  fragments: FragmentAttribution[];
  activeBitId?: number;
  onSelectFragment: (frag: FragmentAttribution) => void;
  pharmacophoreSummary: string;
}

export const FragmentExplainer: React.FC<Props> = ({
  fragments,
  activeBitId,
  onSelectFragment,
  pharmacophoreSummary,
}) => {
  if (!fragments || fragments.length === 0) {
    return (
      <div className="liquid-glass rounded-xl p-4 text-xs text-[#8b949e]">
        No feature attribution data for this molecule.
      </div>
    );
  }

  // Format data for the bar chart
  const chartData = fragments.map((f) => {
    const isRisk = f.direction === "increases_risk";
    const deltaPercent = Number((f.delta * 100).toFixed(1));
    return {
      name: f.fragment_type,
      bitId: f.bit_id,
      delta: deltaPercent,
      isRisk: isRisk,
      rawFragment: f,
    };
  });

  return (
    <div className="liquid-glass rounded-xl p-4 sm:p-5 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 text-xs">
        <div>
          <span className="font-semibold text-[#e6edf3]">
            Substructure Feature Contribution
          </span>
          <span className="text-[#8b949e] block text-[11px]">
            Bar chart showing marginal impact on predicted hERG inhibition risk (&Delta;P). Click any bar or table row to highlight the substructure on the 2D molecule.
          </span>
        </div>
        {activeBitId !== undefined && (
          <span className="text-[#58a6ff] text-[11px] font-mono">
            Active Highlight: Bit #{activeBitId}
          </span>
        )}
      </div>

      {/* Visual Bar Chart */}
      <div className="bg-[#0b0f15]/60 border border-white/10 rounded-lg p-3">
        <div className="text-[11px] font-semibold text-[#8b949e] mb-2 flex items-center justify-between">
          <span>Top Fragment Contribution Bar Chart (&Delta; Probability %)</span>
          <span className="text-[10px]">
            <span className="text-[#f85149] font-medium mr-2">&bull; Increases Risk (+)</span>
            <span className="text-[#3fb950] font-medium">&bull; Decreases Risk (-)</span>
          </span>
        </div>

        <div className="h-48 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 5, right: 20, left: 20, bottom: 5 }}
            >
              <XAxis
                type="number"
                unit="%"
                tick={{ fontSize: 10, fill: "#8b949e" }}
                stroke="rgba(255, 255, 255, 0.1)"
              />
              <YAxis
                type="category"
                dataKey="name"
                width={140}
                tick={{ fontSize: 10, fill: "#e6edf3" }}
                stroke="rgba(255, 255, 255, 0.1)"
              />
              <ReferenceLine x={0} stroke="rgba(255, 255, 255, 0.2)" />
              <Tooltip
                formatter={(value: any, _name: any, item: any) => [
                  `${value > 0 ? "+" : ""}${value}%`,
                  item.payload.isRisk ? "Risk Factor" : "Protective Factor",
                ]}
                labelFormatter={(label) => `Fragment: ${label}`}
                contentStyle={{
                  backgroundColor: "rgba(22, 27, 34, 0.95)",
                  borderColor: "rgba(255, 255, 255, 0.15)",
                  borderRadius: "8px",
                  color: "#e6edf3",
                  fontSize: "11px",
                  backdropFilter: "blur(12px)",
                }}
              />
              <Bar
                dataKey="delta"
                radius={[4, 4, 4, 4]}
                cursor="pointer"
                onClick={(entry: any) => {
                  if (entry && entry.rawFragment) {
                    onSelectFragment(entry.rawFragment);
                  }
                }}
              >
                {chartData.map((entry) => (
                  <Cell
                    key={`cell-${entry.bitId}`}
                    fill={entry.isRisk ? "#f85149" : "#3fb950"}
                    stroke={activeBitId === entry.bitId ? "#58a6ff" : "transparent"}
                    strokeWidth={activeBitId === entry.bitId ? 2 : 0}
                    fillOpacity={activeBitId === undefined || activeBitId === entry.bitId ? 0.9 : 0.4}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Tabular Details */}
      <div className="overflow-x-auto text-xs">
        <table>
          <thead>
            <tr>
              <th>Feature / Substructure</th>
              <th>Impact on Risk</th>
              <th>Mapped Atoms</th>
            </tr>
          </thead>
          <tbody>
            {fragments.map((frag) => {
              const isSelected = activeBitId === frag.bit_id;
              const isRisk = frag.direction === "increases_risk";

              return (
                <tr
                  key={frag.bit_id}
                  onClick={() => onSelectFragment(frag)}
                  className={`cursor-pointer transition-colors ${
                    isSelected
                      ? "bg-[#21262d]/90 font-medium border-l-2 border-[#58a6ff]"
                      : "hover:bg-[#1c2128]/70"
                  }`}
                >
                  <td>
                    <div className="text-[#e6edf3] font-medium">{frag.fragment_type}</div>
                    <div className="text-[11px] text-[#8b949e]">{frag.mechanistic_explanation}</div>
                    {frag.sub_smiles && (
                      <code className="text-[10px] text-[#8b949e]">{frag.sub_smiles}</code>
                    )}
                  </td>
                  <td className="font-mono whitespace-nowrap">
                    <span className={isRisk ? "text-[#ff7b72] font-semibold" : "text-[#7ee787] font-semibold"}>
                      {frag.delta > 0 ? `+${(frag.delta * 100).toFixed(1)}%` : `${(frag.delta * 100).toFixed(1)}%`}
                    </span>
                  </td>
                  <td className="font-mono text-[#8b949e] whitespace-nowrap">
                    [{frag.atom_indices.join(", ")}]
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {pharmacophoreSummary && (
        <div className="text-[11px] text-[#8b949e] pt-1 border-t border-white/10">
          <strong>Pharmacophore Summary:</strong> {pharmacophoreSummary}
        </div>
      )}
    </div>
  );
};
