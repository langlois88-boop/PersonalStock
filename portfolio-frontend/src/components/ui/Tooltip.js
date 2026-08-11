import { useState } from 'react';
import { Info } from 'lucide-react';

/**
 * Small inline "(i)" icon that reveals an explanatory tooltip on hover/focus.
 * Used to explain ML/trading jargon (Sharpe Ratio, confidence, etc.) right
 * where it appears, instead of sending the user to a separate glossary page.
 *
 * `placement="top"` opens the bubble above the icon instead of below --
 * opt-in, default unchanged, so existing call sites keep their exact prior
 * behavior. Added for content living near the bottom of a page (e.g.
 * TickerDetail's ratio grid), where opening below overflowed off the
 * viewport and forced a page scrollbar (reported live, 2026-08-10).
 */
export function InfoTooltip({ text, placement = 'bottom' }) {
  const [open, setOpen] = useState(false);

  if (!text) return null;

  const bubblePosition = placement === 'top' ? 'bottom-full mb-2' : 'top-full mt-2';

  return (
    <span className="relative inline-flex items-center normal-case font-normal">
      <button
        type="button"
        aria-label="Explication"
        className="ml-1 inline-flex text-slate-500 hover:text-indigo-300 focus:text-indigo-300 focus:outline-none"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={(e) => {
          e.stopPropagation();
          setOpen((prev) => !prev);
        }}
      >
        <Info size={13} />
      </button>
      {open && (
        <span
          role="tooltip"
          className={`absolute z-50 left-1/2 -translate-x-1/2 ${bubblePosition} w-56 rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-xs font-normal normal-case leading-snug text-slate-200 shadow-xl`}
        >
          {text}
        </span>
      )}
    </span>
  );
}

/**
 * Wraps a label with an InfoTooltip. Usage: <Term text={GLOSSARY.sharpeRatio}>Sharpe Ratio</Term>
 */
export function Term({ text, children }) {
  return (
    <span className="inline-flex items-center">
      {children}
      <InfoTooltip text={text} />
    </span>
  );
}

export default InfoTooltip;
