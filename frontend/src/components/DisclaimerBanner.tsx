import React from "react";

export const DisclaimerBanner: React.FC = () => {
  return (
    <div className="bg-[#161b22] border-b border-[#30363d] text-[#8b949e] px-4 py-2 text-xs text-center">
      This tool provides computational predictions for research and educational purposes. It is not a clinical diagnostic tool and should not be used as the sole basis for drug-safety decisions.
    </div>
  );
};
