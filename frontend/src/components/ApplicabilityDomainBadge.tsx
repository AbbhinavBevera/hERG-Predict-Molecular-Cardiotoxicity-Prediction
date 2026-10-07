import React from "react";
import type { ApplicabilityDomainResult } from "../types";

interface Props {
  ad: ApplicabilityDomainResult;
}

export const ApplicabilityDomainBadge: React.FC<Props> = ({ ad }) => {
  const isOut = ad.status === "Out-of-Domain";
  const isBorderline = ad.status === "Borderline";

  return (
    <div className="liquid-glass rounded-xl p-4 text-xs space-y-2.5">
      <div className="flex items-center justify-between">
        <span className="font-semibold text-[#e6edf3]">Applicability Domain</span>
        <span
          className={`font-mono font-bold px-2 py-0.5 rounded-md border ${
            isOut
              ? "bg-[#25171c]/80 border-[#f85149]/40 text-[#ff7b72] shadow-[0_0_10px_rgba(248,81,73,0.2)]"
              : isBorderline
              ? "bg-[#272115]/80 border-[#d29922]/40 text-[#e3b341] shadow-[0_0_10px_rgba(210,153,34,0.2)]"
              : "bg-[#14231b]/80 border-[#3fb950]/40 text-[#3fb950] shadow-[0_0_10px_rgba(63,185,80,0.2)]"
          }`}
        >
          {ad.status}
        </span>
      </div>

      <div className="flex items-center justify-between text-[#8b949e] font-mono text-[11px]">
        <span>Max Training Similarity:</span>
        <span className="text-[#e6edf3] font-semibold">{ad.max_tanimoto.toFixed(3)} (Threshold: &ge; {ad.thresholds?.in_domain ?? 0.40})</span>
      </div>

      {ad.warning ? (
        <div className={`p-2.5 rounded-lg border text-xs leading-relaxed ${isOut ? "bg-[#25171c]/90 border-[#f85149]/50 text-[#ff7b72]" : "bg-[#272115]/90 border-[#d29922]/50 text-[#e3b341]"}`}>
          <strong>Warning:</strong> {ad.warning}
        </div>
      ) : (
        <div className="text-[#8b949e] text-[11px] leading-relaxed">
          {ad.explanation}
        </div>
      )}
    </div>
  );
};
