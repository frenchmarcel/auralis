import { useEffect, useRef, useState } from 'react';
import { useTime, type TimeStore } from '../lib/timeline';

const RATES = [0.25, 0.5, 1, 2, 4];

/**
 * Preview clock of the trajectories: slider 0..tMax, play/pause, playback rate.
 * `driver` names an external clock (e.g. a playing render) that moves the time instead.
 */
export function Timeline({
  time,
  tMax,
  onPlayingChange,
  driver,
  className = '',
}: {
  time: TimeStore;
  tMax: number;
  onPlayingChange?: (playing: boolean) => void;
  driver?: string | null;
  className?: string;
}) {
  const t = useTime(time);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(1);
  const rateRef = useRef(rate);
  rateRef.current = rate;
  const tMaxRef = useRef(tMax);
  tMaxRef.current = tMax;
  const notify = useRef(onPlayingChange);
  notify.current = onPlayingChange;

  useEffect(() => {
    notify.current?.(playing);
  }, [playing]);
  useEffect(() => () => notify.current?.(false), []);

  useEffect(() => {
    if (time.get() > tMax) time.set(tMax);
  }, [tMax, time]);

  useEffect(() => {
    if (driver) setPlaying(false);
  }, [driver]);

  useEffect(() => {
    if (!playing) return;
    let id = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      const next = Math.min(tMaxRef.current, time.get() + dt * rateRef.current);
      time.set(next);
      if (next >= tMaxRef.current) {
        setPlaying(false);
        return;
      }
      id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [playing, time]);

  const toggle = () => {
    if (playing) {
      setPlaying(false);
      return;
    }
    if (time.get() >= tMax - 1e-6) time.set(0);
    setPlaying(true);
  };

  return (
    <div className={`timeline-bar ${className}`}>
      <button className={`btn btn-xs tl-play ${playing ? 'on' : ''}`} onClick={toggle} disabled={!!driver || tMax <= 0} title={playing ? 'Pause the preview' : 'Play the trajectories'} aria-label={playing ? 'Pause' : 'Play'}>
        {playing ? '❚❚' : '▶'}
      </button>
      <input
        type="range"
        className="tl-slider"
        min={0}
        max={Math.max(tMax, 0.01)}
        step={0.01}
        value={Math.min(t, tMax)}
        disabled={!!driver}
        onChange={(e) => time.set(Number(e.target.value))}
        aria-label="Preview time"
      />
      <span className="small mono tl-time">
        {t.toFixed(2)} / {tMax.toFixed(2)} s
      </span>
      <select className="tl-rate" value={rate} onChange={(e) => setRate(Number(e.target.value))} aria-label="Playback rate" title="Playback rate" disabled={!!driver}>
        {RATES.map((r) => (
          <option key={r} value={r}>
            {r}×
          </option>
        ))}
      </select>
      {driver && <span className="small muted ellipsis">following {driver}</span>}
    </div>
  );
}
