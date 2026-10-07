import React from "react";
import {
  HeartPulse,
  Atom,
  FileSpreadsheet,
  LineChart,
  BookOpen,
} from "lucide-react";

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab }) => {
  const tabs = [
    { id: "predict", label: "Predict Molecule", icon: Atom },
    { id: "batch", label: "Batch CSV", icon: FileSpreadsheet },
    { id: "gnn", label: "Model Benchmark", icon: LineChart },
    { id: "methodology", label: "Methodology", icon: BookOpen },
  ];

  return (
    <header className="bg-[#161b22]/70 backdrop-blur-md border-b border-white/10 w-full sticky top-0 z-50">
      <div className="w-full px-4 sm:px-6 flex items-center justify-between h-14">
        {/* Far Left: Logo & Heading */}
        <div
          onClick={() => setActiveTab("predict")}
          className="flex items-center gap-2 cursor-pointer select-none group"
        >
          <div className="w-8 h-8 rounded-lg bg-[#21262d]/80 border border-white/10 text-[#58a6ff] flex items-center justify-center transition-all duration-300 group-hover:border-[#58a6ff]/40 group-hover:shadow-[0_0_15px_rgba(88,166,255,0.25)]">
            <HeartPulse className="w-4 h-4 transition-transform duration-300 group-hover:scale-110" />
          </div>
          <span className="font-bold text-sm sm:text-base text-[#e6edf3] tracking-tight">
            hERG Risk Predictor
          </span>
        </div>

        {/* Right: Nav Tabs with Icons */}
        <nav className="flex items-center space-x-1 sm:space-x-2">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all duration-200 cursor-pointer ${
                  isActive
                    ? "bg-[#21262d]/90 text-[#e6edf3] border border-white/15 shadow-[0_0_12px_rgba(88,166,255,0.15)]"
                    : "text-[#8b949e] hover:text-[#e6edf3] hover:bg-[#21262d]/40 border border-transparent"
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};
