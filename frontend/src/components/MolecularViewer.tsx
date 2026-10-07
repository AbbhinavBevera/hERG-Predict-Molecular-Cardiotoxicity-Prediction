import React from "react";
import type { MolecularDescriptors } from "../types";

interface Props {
  svg: string;
  descriptors: MolecularDescriptors;
  highlightedCount: number;
  onResetHighlight: () => void;
  activeFragmentName?: string;
}

export const MolecularViewer: React.FC<Props> = ({
  svg,
  descriptors,
  highlightedCount,
  onResetHighlight,
  activeFragmentName,
}) => {
  return (
    <div className="liquid-glass rounded-xl p-4 flex flex-col justify-between">
      <div>
        <div className="flex items-center justify-between pb-2 mb-2 border-b border-white/10 text-xs">
          <span className="font-semibold text-[#e6edf3]">2D Structure</span>
          {highlightedCount > 0 && (
            <button
              onClick={onResetHighlight}
              className="text-[#58a6ff] hover:underline text-xs cursor-pointer"
            >
              Clear highlight
            </button>
          )}
        </div>

        {activeFragmentName && (
          <div className="mb-2 text-xs text-[#8b949e]">
            Highlighting: <strong className="text-[#e6edf3]">{activeFragmentName}</strong> ({highlightedCount} atoms)
          </div>
        )}

        <div className="molecule-svg-container bg-[#0b0f15]/80 backdrop-blur-xs border border-white/10 rounded-lg flex items-center justify-center p-3 min-h-[220px]">
          {svg ? (
            <div
              className="w-full flex items-center justify-center"
              dangerouslySetInnerHTML={{ __html: svg }}
            />
          ) : (
            <div className="text-[#8b949e] text-xs">No molecule loaded.</div>
          )}
        </div>
      </div>

      <div className="mt-3 pt-2.5 border-t border-white/10 flex items-center justify-between text-xs text-[#8b949e] font-mono">
        <span className="text-[#e6edf3]">{descriptors.formula || "—"}</span>
        <span>Atoms: {descriptors.heavy_atoms} | Rings: {descriptors.aromatic_rings}</span>
      </div>
    </div>
  );
};
