"use client";

const SUGGESTIONS = [
  "What are the eligibility criteria to join?",
  "What is the fee structure for Jan 2026?",
  "How is grading calculated?",
  "What courses are in the Foundation level?",
  "Can I exit with a Diploma?",
  "What happens if I fail a qualifier exam?",
];

interface SuggestionChipsProps {
  dark: boolean;
  onSelect: (q: string) => void;
}

export default function SuggestionChips({ dark, onSelect }: SuggestionChipsProps) {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-6 px-4 py-10 animate-fadeUp">
      <h1 className={`text-2xl sm:text-3xl font-semibold text-center leading-snug ${dark ? "text-white/90" : "text-gray-800"}`}>
        What can I help you with<br />
        about <span className="text-[#c8102e]">IITM BS DS Programme?</span>
      </h1>
      <p className={`text-sm text-center max-w-md leading-relaxed ${dark ? "text-white/45" : "text-gray-500"}`}>
        Ask anything about courses, fees, eligibility, grading, or deadlines — answered from official documents.
      </p>
      <div className="flex flex-wrap gap-2 justify-center max-w-xl">
        {SUGGESTIONS.map(s => (
          <button
            key={s}
            onClick={() => onSelect(s)}
            className={`px-4 py-2 text-sm rounded-full border transition-all active:scale-95 hover:-translate-y-0.5 ${
              dark
                ? "bg-white/5 border-white/10 text-white/50 hover:bg-white/10 hover:text-white/80 hover:border-[#c8102e]"
                : "bg-black/4 border-black/10 text-gray-500 hover:bg-black/8 hover:text-gray-700 hover:border-[#c8102e]"
            }`}
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}