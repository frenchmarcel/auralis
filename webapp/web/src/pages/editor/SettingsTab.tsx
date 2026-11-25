import { useEffect, useState } from 'react';
import { api, BANDS_HZ, type BackgroundNoise, type HeadInfo, type SettingsMode, type SimulationSettings, type SolverChoice, type SpeechEffort, type Validation } from '../../api';
import { NumField, Segmented, Toggle } from '../../components/ui';
import { bandLabel } from '../../lib/format';
import { fmtHz, fmtNum } from '../../lib/format';

const MODES: { value: SettingsMode; title: string; text: string }[] = [
  { value: 'survey', title: 'Survey', text: 'Geometrical solver only (image sources + rays). Fast; IR length from the estimated reverberation time.' },
  { value: 'default', title: 'Default', text: 'Hybrid: wave solver below the proposed transition frequency, geometrical above. Needs a closed model, or an open scene.' },
  { value: 'advanced', title: 'Advanced', text: 'Full control: transition frequency, IR length, solvers, image-source order, rays and engine options.' },
];

const OCTAVE_TICKS = [63, 125, 250, 500, 1000, 2000];
const DEFAULT_GRID_RAYS = 50000;
/** Decay that covers T30 (evaluated from -5 to -35 dB); the server adds 10 dB of margin. */
const DEFAULT_STOP_DECAY_DB = 35;
const F_MIN = 63 / Math.SQRT2;
const F_MAX = 2000 * Math.SQRT2;
const SLIDER_STEPS = 1000;

/** NC curves (ANSI S12.2), 125 Hz–8 kHz: only to prefill a custom spectrum; the server's table is authoritative. */
const NC_PREFILL: Record<number, number[]> = {
  15: [36, 29, 22, 17, 14, 12, 11], 20: [40, 33, 26, 22, 19, 17, 16], 25: [44, 37, 31, 27, 24, 22, 21],
  30: [48, 41, 35, 31, 29, 28, 27], 35: [52, 45, 40, 36, 34, 33, 32], 40: [56, 50, 45, 41, 39, 38, 37],
  45: [60, 54, 49, 46, 44, 43, 42], 50: [64, 58, 54, 51, 49, 48, 47], 55: [67, 62, 58, 56, 54, 53, 52],
  60: [71, 67, 63, 61, 59, 58, 57], 65: [75, 71, 68, 66, 64, 63, 62], 70: [79, 75, 72, 71, 70, 69, 68],
};
const NC_LEVELS = Object.keys(NC_PREFILL).map(Number);

function noiseKind(n: BackgroundNoise | undefined): string {
  if (Array.isArray(n)) return 'custom';
  if (!n || n === 'none') return 'none';
  return n.toUpperCase();
}

const toSlider = (f: number) => Math.round((Math.log(f / F_MIN) / Math.log(F_MAX / F_MIN)) * SLIDER_STEPS);
const fromSlider = (s: number) => F_MIN * Math.pow(F_MAX / F_MIN, s / SLIDER_STEPS);

export function SettingsTab({
  settings,
  onChange,
  validation,
  locked,
}: {
  settings: SimulationSettings;
  onChange: (patch: Partial<SimulationSettings>) => void;
  validation: Validation | null;
  locked: boolean;
}) {
  const waveOk = validation ? validation.wave_available : true;
  const notClosed = validation?.watertight === false;
  const openScene = !!validation?.open_scene;
  // closed only at the voxel scale: may be a hill under open-air seats rather than a room
  const voxelClosed = validation?.watertight === true && validation?.closure_method === 'voxels';
  const pointsOutside =
    !!validation && [...validation.sources, ...validation.receivers].some((p) => p.reasons.some((r) => r.includes('not in the air volume')));
  const askOpen = notClosed || (voxelClosed && (pointsOutside || !!settings.open_scene));
  const suggested = validation?.suggested_transition_hz ?? null;
  const advanced = settings.mode === 'advanced';
  const usesWave = settings.mode === 'default' || (advanced && settings.solvers !== 'geometrical');
  const transitionMatters = advanced && settings.solvers !== 'geometrical';
  const tVal = settings.transition_hz ?? suggested ?? 500;

  return (
    <div className="tab-content settings-tab">
      <fieldset disabled={locked}>
        <div className="info-strip">
          <span>
            Wave solver:{' '}
            {validation ? (
              waveOk ? (
                <b className="text-ok">available</b>
              ) : (
                <b className="text-warn">unavailable (model not closed)</b>
              )
            ) : (
              '…'
            )}
          </span>
          <span>
            Suggested transition: <b>{suggested !== null ? fmtHz(suggested) : '–'}</b>
          </span>
        </div>
        {askOpen && (
          <div className="open-scene-box">
            <div className="section-title">
              {notClosed ? 'This model is not closed. What is it?' : 'Some positions lie outside the enclosed volume. What is this model?'}
            </div>
            <label className="radio">
              <input type="radio" name="open_scene" checked={!!settings.open_scene} onChange={() => onChange({ open_scene: true })} />
              <span>An open scene</span>
              <span className="muted small">an object in free field (panel, wedge, seats…): the wave solver runs with absorbing boundaries around it</span>
            </label>
            <label className="radio">
              <input type="radio" name="open_scene" checked={!settings.open_scene} onChange={() => onChange({ open_scene: false })} />
              <span>{notClosed ? 'A room with holes' : 'A closed room'}</span>
              <span className="muted small">
                {notClosed
                  ? 'energy would leak through the holes and shorten the reverberation: fix the open edges and re-import, or use the geometrical solver only'
                  : 'keep it closed and move the sources/receivers inside the room'}
              </span>
            </label>
          </div>
        )}
        {!waveOk && usesWave && (
          <div className="warn-box small">
            The current settings use the wave solver, but the model is not closed. Declare it an open scene above, or switch to Survey or to the geometrical solver only.
          </div>
        )}

        <div className="mode-cards" role="radiogroup" aria-label="Simulation mode">
          {MODES.map((m) => {
            const disabled = m.value === 'default' && !waveOk;
            return (
              <label key={m.value} className={`mode-card ${settings.mode === m.value ? 'active' : ''} ${disabled ? 'disabled' : ''}`}>
                <input
                  type="radio"
                  name="mode"
                  value={m.value}
                  checked={settings.mode === m.value}
                  disabled={disabled}
                  onChange={() => onChange({ mode: m.value })}
                />
                <span className="mode-title">{m.title}</span>
                <span className="mode-text">{disabled ? 'Unavailable: the wave solver needs a closed model, or an open scene.' : m.text}</span>
              </label>
            );
          })}
        </div>

        {settings.mode === 'survey' && (
          <dl className="kv small">
            <dt>Solvers</dt>
            <dd>geometrical only</dd>
            <dt>IR length</dt>
            <dd>automatic (from the estimated reverberation time)</dd>
          </dl>
        )}
        {settings.mode === 'default' && (
          <dl className="kv small">
            <dt>Transition</dt>
            <dd>{suggested !== null ? `${fmtHz(suggested)} (proposed by the engine)` : 'automatic'}</dd>
            <dt>IR length</dt>
            <dd>automatic</dd>
          </dl>
        )}

        {advanced && (
          <>
            <div className="section-title">Solvers</div>
            <div className="radio-col">
              {(
                [
                  ['both', 'Both (hybrid)', 'Wave below the transition, geometrical above.'],
                  ['wave', 'Wave solver only', 'Results up to the transition frequency only.'],
                  ['geometrical', 'Geometrical only', 'Image sources + rays, all bands.'],
                ] as [SolverChoice, string, string][]
              ).map(([v, label, help]) => {
                const disabled = v !== 'geometrical' && !waveOk;
                return (
                  <label key={v} className={`radio ${disabled ? 'disabled' : ''}`} title={disabled ? 'Needs a closed model, or an open scene' : help}>
                    <input type="radio" name="solvers" checked={settings.solvers === v} disabled={disabled} onChange={() => onChange({ solvers: v })} />
                    <span>{label}</span>
                    <span className="muted small">{disabled ? 'needs a closed model, or an open scene' : help}</span>
                  </label>
                );
              })}
            </div>

            <div className={`section-title ${transitionMatters ? '' : 'dim'}`}>
              Transition frequency
              <label className="check inline">
                <input
                  type="checkbox"
                  checked={settings.transition_hz === null}
                  disabled={!transitionMatters}
                  onChange={(e) => onChange({ transition_hz: e.target.checked ? null : Math.round(tVal) })}
                />{' '}
                auto
              </label>
            </div>
            <div className={`transition ${!transitionMatters || settings.transition_hz === null ? 'disabled' : ''}`}>
              <input
                type="range"
                min={0}
                max={SLIDER_STEPS}
                value={toSlider(Math.min(F_MAX, Math.max(F_MIN, tVal)))}
                disabled={!transitionMatters || settings.transition_hz === null}
                onChange={(e) => onChange({ transition_hz: Math.round(fromSlider(Number(e.target.value))) })}
                aria-label="Transition frequency"
              />
              <div className="ticks">
                {OCTAVE_TICKS.map((f) => {
                  const cutoff = Math.round(f * Math.SQRT2);
                  const pos = (toSlider(f) / SLIDER_STEPS) * 100;
                  const on = cutoff <= tVal + 0.5;
                  return (
                    <button
                      key={f}
                      type="button"
                      className={`tick ${on ? 'on' : ''}`}
                      style={{ left: `${pos}%` }}
                      disabled={!transitionMatters}
                      title={`Include the ${f >= 1000 ? `${f / 1000}k` : f} Hz octave in the wave band: transition = ${cutoff} Hz`}
                      onClick={() => onChange({ transition_hz: cutoff })}
                    >
                      {f >= 1000 ? `${f / 1000}k` : f}
                    </button>
                  );
                })}
              </div>
              <div className="row gap">
                <NumField
                  value={settings.transition_hz}
                  allowNull
                  min={20}
                  max={4000}
                  step={10}
                  digits={0}
                  placeholder={suggested !== null ? String(Math.round(suggested)) : 'auto'}
                  disabled={!transitionMatters}
                  onChange={(v) => onChange({ transition_hz: v === null ? null : Math.round(v) })}
                  width={80}
                />
                <span className="unit">Hz</span>
                {suggested !== null && settings.transition_hz !== null && Math.round(suggested) !== settings.transition_hz && (
                  <button type="button" className="link small" disabled={!transitionMatters} onClick={() => onChange({ transition_hz: Math.round(suggested) })}>
                    use suggested ({Math.round(suggested)} Hz)
                  </button>
                )}
              </div>
              <div className="muted small">Wave-solver cost grows roughly with f⁴: doubling the transition costs about 16×.</div>
            </div>

            <StopCriterion settings={settings} onChange={onChange} />

            <div className="section-title">Geometrical solver</div>
            <div className="form-grid compact">
              <label>Image-source order</label>
              <NumField value={settings.ism_order} min={1} max={10} step={1} digits={0} onChange={(v) => onChange({ ism_order: Math.round(v ?? 2) })} width={80} />
              <label>Rays</label>
              <div className="row gap">
                <NumField value={settings.n_rays} min={1000} max={1500000} step={10000} digits={0} onChange={(v) => onChange({ n_rays: Math.round(v ?? 200000) })} width={100} />
                <span className="muted small">1 000 – 1 500 000</span>
              </div>
              <label>Diffraction</label>
              <Toggle checked={settings.diffraction} onChange={(v) => onChange({ diffraction: v })} label={settings.diffraction ? 'on' : 'off'} />
              <label title="Rays traced for each point of the receiver grids (surface receivers)">Grid rays per point</label>
              <div className="row gap">
                <NumField
                  value={settings.grid_rays ?? DEFAULT_GRID_RAYS}
                  min={1000}
                  max={2000000}
                  step={10000}
                  digits={0}
                  onChange={(v) => onChange({ grid_rays: Math.round(v ?? DEFAULT_GRID_RAYS) })}
                  width={100}
                  ariaLabel="Rays per receiver-grid point"
                />
                <span className="muted small">1 000 – 2 000 000 (receiver grids only)</span>
              </div>
            </div>

            <div className="section-title">Scattering</div>
            <div className="form-grid compact">
              <label title="Christensen & Rindel (Forum Acusticum 2005): adds, per reflection, the scattering due to the finite size of each surface (size, incidence, distances) to the material's own">
                Surface-size scattering
              </label>
              <Toggle checked={!!settings.rbs_scattering} onChange={(v) => onChange({ rbs_scattering: v })} label={settings.rbs_scattering ? 'on' : 'off'} />
            </div>
            <div className="muted small">
              On by default: validated on the three BRAS rooms, and it also counts the image sources at their specular share only (no double count with the scattered rays). The materials&apos; scattering should describe their roughness only (e.g. 0.05 for smooth walls).
            </div>

            <div className="section-title">Wave solver</div>
            <div className="form-grid compact">
              <label>Conformal correction</label>
              {openScene ? (
                <span className="muted">off (open scene)</span>
              ) : (
                <Toggle checked={settings.conformal} onChange={(v) => onChange({ conformal: v })} label={settings.conformal ? 'on' : 'off'} />
              )}
            </div>
            <div className="warn-box small">
              {openScene
                ? 'Disabled for open scenes: it degrades them (measured on BRAS RS6/RS7) and doubles the area of thin panels.'
                : 'Known to degrade results on open scenes with sharp-edged objects; keep it on for closed rooms.'}
            </div>
          </>
        )}

        <div className="section-title">Air</div>
        <div className="form-grid compact">
          <label>Temperature</label>
          <div className="row gap">
            <NumField value={settings.temperature_c} min={-30} max={50} step={0.5} digits={1} onChange={(v) => onChange({ temperature_c: v ?? 20 })} width={80} />
            <span className="unit">°C</span>
          </div>
          <label>Humidity</label>
          <div className="row gap">
            <NumField value={settings.humidity_percent} min={0} max={100} step={5} digits={0} onChange={(v) => onChange({ humidity_percent: v ?? 50 })} width={80} />
            <span className="unit">%</span>
          </div>
        </div>
        {validation && (
          <div className="muted small">
            Model volume {validation.volume_m3 !== null ? `${fmtNum(validation.volume_m3, 1)} m³` : '– (open)'}
          </div>
        )}

        <HeadSettings settings={settings} onChange={onChange} />

        <StiSettings settings={settings} onChange={onChange} />
      </fieldset>
    </div>
  );
}

const MODEL_HEAD: HeadInfo = { id: 'model', name: 'Spherical head (model)', available: true, credit: null };

/** Listener head of the binaural rendering (spherical model or a measured set). */
function HeadSettings({ settings, onChange }: { settings: SimulationSettings; onChange: (patch: Partial<SimulationSettings>) => void }) {
  const [heads, setHeads] = useState<HeadInfo[]>([MODEL_HEAD]);
  useEffect(() => {
    let live = true;
    api.getHeads().then((h) => live && setHeads(h)).catch(() => undefined);
    return () => {
      live = false;
    };
  }, []);
  const current = settings.head ?? 'model';
  const info = heads.find((h) => h.id === current);
  return (
    <>
      <div className="section-title">Listener head</div>
      <div className="form-grid compact">
        <label>Binaural</label>
        <select value={current} onChange={(e) => onChange({ head: e.target.value })} aria-label="Listener head">
          {heads.map((h) => (
            <option key={h.id} value={h.id} disabled={!h.available && h.id !== current}>
              {h.name}
              {h.available ? '' : ' – not installed'}
            </option>
          ))}
        </select>
      </div>
      <div className="muted small">
        {current === 'model'
          ? 'Sphere without pinnae: sounds tend to stay inside the head, front and back get confused.'
          : 'Measured head and torso: sound outside the head, front/back cues. Head tracking uses it at once; the fixed-head IRs need a new run.'}
        {info?.credit ? ` ${info.credit}.` : ''}
      </div>
    </>
  );
}

function StiSettings({ settings, onChange }: { settings: SimulationSettings; onChange: (patch: Partial<SimulationSettings>) => void }) {
  const effort: SpeechEffort = settings.speech_effort === 'raised' ? 'raised' : 'normal';
  const noise = settings.background_noise ?? 'none';
  const kind = noiseKind(noise);
  const custom = Array.isArray(noise) ? noise : null;
  const setKind = (k: string) => {
    if (k === 'none') onChange({ background_noise: 'none' });
    else if (k === 'custom') {
      const from = typeof noise === 'string' ? NC_PREFILL[Number(noise.toUpperCase().replace('NC-', ''))] : undefined;
      onChange({ background_noise: [...(from ?? NC_PREFILL[35])] });
    } else onChange({ background_noise: k });
  };
  return (
    <>
      <div className="section-title">Speech intelligibility (STI)</div>
      <div className="form-grid compact">
        <label>Speech effort</label>
        <div className="row gap">
          <Segmented<SpeechEffort>
            value={effort}
            onChange={(v) => onChange({ speech_effort: v })}
            options={[
              { value: 'normal', label: 'Normal', title: '60 dB(A) at 1 m (IEC 60268-16 male speech spectrum)' },
              { value: 'raised', label: 'Raised', title: '66 dB(A) at 1 m (IEC 60268-16 male speech spectrum)' },
            ]}
          />
          <span className="muted small">{effort === 'raised' ? '66' : '60'} dB(A) at 1 m</span>
        </div>
        <label>Background noise</label>
        <select value={kind} onChange={(e) => setKind(e.target.value)} aria-label="Background noise">
          <option value="none">None</option>
          {NC_LEVELS.map((n) => (
            <option key={n} value={`NC-${n}`}>
              NC-{n}
            </option>
          ))}
          <option value="custom">Custom (octave levels)…</option>
        </select>
      </div>
      {custom && (
        <div className="noise-grid" role="group" aria-label="Background noise per octave band (dB)">
          {BANDS_HZ.map((f, i) => (
            <label key={f} className="noise-cell">
              <span className="muted small">{bandLabel(f)}</span>
              <NumField
                value={custom[i] ?? 0}
                min={-20}
                max={130}
                step={1}
                digits={1}
                ariaLabel={`Noise level at ${bandLabel(f)} Hz`}
                onChange={(v) => {
                  const n = [...custom];
                  n[i] = v ?? 0;
                  onChange({ background_noise: n });
                }}
                width={50}
              />
            </label>
          ))}
          <span className="unit">dB SPL per octave</span>
        </div>
      )}
      <div className="muted small">
        Used only by the STI (speech at the listener = this effort&apos;s spectrum at 1 m + G − 20 dB; noise masks it). Changing it marks the results stale: re-run to update the STI.
      </div>
    </>
  );
}

type StopKind = 'auto' | 'length' | 'decay';

/** Advanced: IR length = automatic, a fixed length, or long enough for a decay of N dB in every band. */
function StopCriterion({ settings, onChange }: { settings: SimulationSettings; onChange: (patch: Partial<SimulationSettings>) => void }) {
  const kind: StopKind = settings.ir_length_s ? 'length' : settings.stop_decay_db ? 'decay' : 'auto';
  return (
    <>
      <div className="section-title">Stop criterion (IR length)</div>
      <div className="radio-col stop-criterion">
        <label className="radio">
          <input type="radio" name="stop" checked={kind === 'auto'} onChange={() => onChange({ ir_length_s: null, stop_decay_db: null })} />
          <span>Automatic</span>
          <span className="muted small">from the estimated reverberation time (500 Hz – 1 kHz)</span>
        </label>
        <label className="radio">
          <input type="radio" name="stop" checked={kind === 'length'} onChange={() => onChange({ ir_length_s: 1.5, stop_decay_db: null })} />
          <span className="row gap">
            IR length
            <NumField
              value={settings.ir_length_s}
              allowNull
              min={0.05}
              max={30}
              step={0.1}
              digits={2}
              placeholder="1.5"
              disabled={kind !== 'length'}
              onChange={(v) => onChange({ ir_length_s: v, stop_decay_db: null })}
              width={70}
              ariaLabel="IR length (s)"
            />
            <span className="unit">s</span>
          </span>
        </label>
        <label className="radio">
          <input type="radio" name="stop" checked={kind === 'decay'} onChange={() => onChange({ ir_length_s: null, stop_decay_db: DEFAULT_STOP_DECAY_DB })} />
          <span className="row gap">
            Decay of
            <NumField
              value={settings.stop_decay_db ?? null}
              allowNull
              min={20}
              max={90}
              step={5}
              digits={0}
              placeholder={String(DEFAULT_STOP_DECAY_DB)}
              disabled={kind !== 'decay'}
              onChange={(v) => onChange({ ir_length_s: null, stop_decay_db: v === null ? null : Math.round(v) })}
              width={60}
              ariaLabel="Decay (dB)"
            />
            <span className="unit">dB</span>
          </span>
          <span className="muted small">20 – 90 dB in every octave band (Eyring per band, + 10 dB margin, 0.3 – 8 s); 35 dB covers T30</span>
        </label>
      </div>
    </>
  );
}
