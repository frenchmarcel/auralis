import { useEffect, useRef, useState, type ReactNode } from 'react';
import type { JobStatus, SimulationStatus } from '../api';

const modalStack: number[] = [];
let modalSeq = 0;

export function anyModalOpen(): boolean {
  return modalStack.length > 0;
}

export function Modal({
  title,
  onClose,
  children,
  width = 560,
  className = '',
  closeOnBackdrop = true,
}: {
  title: ReactNode;
  onClose: () => void;
  children: ReactNode;
  width?: number | string;
  className?: string;
  closeOnBackdrop?: boolean;
}) {
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;
  useEffect(() => {
    const id = ++modalSeq;
    modalStack.push(id);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && modalStack[modalStack.length - 1] === id) {
        e.preventDefault();
        e.stopImmediatePropagation();
        onCloseRef.current();
      }
    };
    window.addEventListener('keydown', onKey, true);
    return () => {
      window.removeEventListener('keydown', onKey, true);
      const i = modalStack.indexOf(id);
      if (i >= 0) modalStack.splice(i, 1);
    };
  }, []);
  return (
    <div
      className="modal-overlay"
      onMouseDown={(e) => {
        if (closeOnBackdrop && e.target === e.currentTarget) onClose();
      }}
    >
      <div className={`modal ${className}`} style={{ width }} role="dialog" aria-modal="true">
        <div className="modal-header">
          <h2>{title}</h2>
          <button className="icon-btn" aria-label="Close" onClick={onClose}>
            ×
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

/**
 * Numeric input that keeps the raw text while editing and commits a parsed,
 * clamped number on blur / Enter. Arrow keys step the value.
 */
export function NumField({
  value,
  onChange,
  min,
  max,
  step = 0.1,
  digits = 3,
  allowNull = false,
  placeholder,
  disabled,
  title,
  className = '',
  ariaLabel,
  width,
}: {
  value: number | null;
  onChange: (v: number | null) => void;
  min?: number;
  max?: number;
  step?: number;
  digits?: number;
  allowNull?: boolean;
  placeholder?: string;
  disabled?: boolean;
  title?: string;
  className?: string;
  ariaLabel?: string;
  width?: number | string;
}) {
  const fmt = (v: number | null) => (v === null || v === undefined ? '' : String(Number(v.toFixed(digits))));
  const [text, setText] = useState(fmt(value));
  const [editing, setEditing] = useState(false);
  useEffect(() => {
    if (!editing) setText(fmt(value));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, editing]);

  const clamp = (v: number) => {
    let r = v;
    if (min !== undefined) r = Math.max(min, r);
    if (max !== undefined) r = Math.min(max, r);
    return r;
  };
  const commit = (raw: string) => {
    const t = raw.trim().replace(',', '.');
    if (t === '') {
      if (allowNull) onChange(null);
      else setText(fmt(value));
      return;
    }
    const v = Number(t);
    if (!Number.isFinite(v)) {
      setText(fmt(value));
      return;
    }
    const c = clamp(v);
    setText(fmt(c));
    if (c !== value) onChange(c);
  };
  return (
    <input
      type="text"
      inputMode="decimal"
      className={`num ${className}`}
      style={width !== undefined ? { width } : undefined}
      value={text}
      placeholder={placeholder}
      disabled={disabled}
      title={title}
      aria-label={ariaLabel}
      onFocus={(e) => {
        setEditing(true);
        e.target.select();
      }}
      onChange={(e) => setText(e.target.value)}
      onBlur={(e) => {
        setEditing(false);
        commit(e.target.value);
      }}
      onKeyDown={(e) => {
        if (e.key === 'Enter') {
          commit((e.target as HTMLInputElement).value);
          (e.target as HTMLInputElement).blur();
        } else if (e.key === 'Escape') {
          setText(fmt(value));
          setEditing(false);
          (e.target as HTMLInputElement).blur();
          e.stopPropagation();
        } else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
          e.preventDefault();
          const base = Number(text.replace(',', '.'));
          const cur = Number.isFinite(base) && text.trim() !== '' ? base : value ?? 0;
          const mult = e.shiftKey ? 10 : 1;
          const next = clamp(Number((cur + (e.key === 'ArrowUp' ? step : -step) * mult).toFixed(digits)));
          setText(fmt(next));
          onChange(next);
        }
      }}
    />
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  disabled,
  size = 'sm',
}: {
  value: T;
  options: { value: T; label: ReactNode; title?: string; disabled?: boolean }[];
  onChange: (v: T) => void;
  disabled?: boolean;
  size?: 'sm' | 'md';
}) {
  return (
    <div className={`segmented segmented-${size}`} role="radiogroup">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          className={o.value === value ? 'active' : ''}
          disabled={disabled || o.disabled}
          title={o.title}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Toggle({
  checked,
  onChange,
  label,
  disabled,
  title,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label?: ReactNode;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <label className={`toggle ${disabled ? 'disabled' : ''}`} title={title}>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      <span className="toggle-track">
        <span className="toggle-thumb" />
      </span>
      {label !== undefined && <span className="toggle-label">{label}</span>}
    </label>
  );
}

export function Spinner({ size = 14 }: { size?: number }) {
  return <span className="spinner" style={{ width: size, height: size }} aria-label="Loading" />;
}

export function ErrorBox({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="error-box" role="alert">
      <span className="error-icon">!</span>
      <span className="error-text">{message}</span>
      {onRetry && (
        <button className="btn btn-sm" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty-state">
      <div className="empty-title">{title}</div>
      {children && <div className="empty-body">{children}</div>}
    </div>
  );
}

const STATUS_LABEL: Record<SimulationStatus, string> = {
  draft: 'Draft',
  queued: 'Queued',
  running: 'Running',
  done: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

export function StatusBadge({ status, stale }: { status: SimulationStatus | JobStatus; stale?: boolean }) {
  return (
    <span className={`status status-${status}`}>
      <span className="status-dot" />
      {STATUS_LABEL[status] ?? status}
      {stale && <span className="status-stale" title="Edited after the last run: results are stale">stale</span>}
    </span>
  );
}

export function ProgressBar({ value, status }: { value: number; status?: string }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className={`progress ${status ? `progress-${status}` : ''}`}>
      <div className="progress-fill" style={{ width: `${pct}%` }} />
    </div>
  );
}

export interface MenuItem {
  label: string;
  onClick: () => void;
  danger?: boolean;
  disabled?: boolean;
  title?: string;
}

/** Small dropdown menu opened from a "⋯" button. */
export function Menu({ items, label = '⋯', title = 'More actions' }: { items: MenuItem[]; label?: ReactNode; title?: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopImmediatePropagation();
        setOpen(false);
      }
    };
    window.addEventListener('mousedown', onDown);
    window.addEventListener('keydown', onKey, true);
    return () => {
      window.removeEventListener('mousedown', onDown);
      window.removeEventListener('keydown', onKey, true);
    };
  }, [open]);
  return (
    <div className="menu" ref={ref}>
      <button
        className="icon-btn"
        title={title}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={(e) => {
          e.stopPropagation();
          setOpen((o) => !o);
        }}
      >
        {label}
      </button>
      {open && (
        <div className="menu-list" role="menu">
          {items.map((it) => (
            <button
              key={it.label}
              role="menuitem"
              className={`menu-item ${it.danger ? 'danger' : ''}`}
              disabled={it.disabled}
              title={it.title}
              onClick={(e) => {
                e.stopPropagation();
                setOpen(false);
                it.onClick();
              }}
            >
              {it.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function EyeButton({ visible, onToggle, title }: { visible: boolean; onToggle: () => void; title?: string }) {
  return (
    <button
      className={`icon-btn eye ${visible ? '' : 'off'}`}
      title={title ?? (visible ? 'Hide' : 'Show')}
      aria-pressed={!visible}
      onClick={(e) => {
        e.stopPropagation();
        onToggle();
      }}
    >
      <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
        {visible ? (
          <>
            <path d="M1 8s2.6-5 7-5 7 5 7 5-2.6 5-7 5-7-5-7-5z" fill="none" stroke="currentColor" strokeWidth="1.3" />
            <circle cx="8" cy="8" r="2.2" fill="currentColor" />
          </>
        ) : (
          <>
            <path d="M1 8s2.6-5 7-5 7 5 7 5-2.6 5-7 5-7-5-7-5z" fill="none" stroke="currentColor" strokeWidth="1.3" opacity="0.5" />
            <path d="M2 14L14 2" stroke="currentColor" strokeWidth="1.4" />
          </>
        )}
      </svg>
    </button>
  );
}

/** True when keyboard focus is in a text-entry element (shortcuts must not fire). */
export function isTypingTarget(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false;
  const tag = el.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
}
