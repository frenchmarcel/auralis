import type { Orientation, Path, PathValidity, Vec3 } from '../../api';
import { NumField, Toggle } from '../../components/ui';
import { fmtNum } from '../../lib/format';
import { isSmooth, localPathInfo, validityMatches } from '../../lib/smoothPath';
import {
  deleteWaypoint,
  insertWaypointAfter,
  makePath,
  MAX_WAYPOINTS,
  pathStats,
  setSegmentSpeed,
  setWaypointPosition,
  setWaypointTime,
  timeBounds,
  type Movable,
} from '../../lib/trajectory';

const ORIENTATIONS: { value: Orientation; label: string; title: string }[] = [
  { value: 'fixed', label: 'Fixed', title: 'Keeps the azimuth / elevation set above' },
  { value: 'follow', label: 'Follows the path', title: 'Azimuth along the direction of travel (corners smoothed over ±0.25 s), elevation kept' },
  { value: 'target', label: 'Aims at a target', title: 'Always faces another source or receiver, which may move too' },
];

const SMOOTH_TITLE =
  'Smooth: a C1 curve through every waypoint at its time (corners rounded, the point really stops at a pause). It can swing out of the straight segments near corners; the checks follow the true curve.';

/**
 * Trajectory of one source or receiver. Collapsed (row not selected): a summary.
 * Expanded: orientation, interpolation and the waypoint table (x, y, z, t, speed of the incoming segment).
 */
export function TrajectorySection({
  item,
  targets,
  validity,
  expanded,
  selectedWaypoint,
  onSelectWaypoint,
  onPath,
  drawing,
  onDraw,
  locked,
}: {
  item: Movable;
  /** The other sources and receivers (possible targets). */
  targets: { id: string; name: string }[];
  validity?: PathValidity;
  expanded: boolean;
  selectedWaypoint: number | null;
  onSelectWaypoint: (index: number | null) => void;
  onPath: (path: Path | null) => void;
  drawing: boolean;
  onDraw: (on: boolean) => void;
  locked: boolean;
}) {
  const path = item.path ?? null;
  if (!path) {
    if (!expanded) return null;
    return (
      <div className="traj">
        <div className="traj-head">
          <span className="traj-title">Trajectory</span>
          <span className="small muted">static</span>
          <div className="grow" />
          <button className="btn btn-xs" disabled={locked} onClick={() => onPath(makePath(item))} title="Start here at t = 0 s, then walk 2 m along the azimuth at 1.4 m/s">
            Make it move
          </button>
          <button
            className={`btn btn-xs ${drawing ? 'btn-on' : ''}`}
            disabled={locked}
            onClick={(e) => {
              e.stopPropagation();
              onDraw(!drawing);
            }}
            title="Click surfaces in the 3D view to add waypoints"
          >
            {drawing ? 'Drawing…' : 'Draw path'}
          </button>
        </div>
      </div>
    );
  }

  const st = pathStats(path);
  const smooth = isSmooth(path);
  // Length and top speed as the server checked them (along the curve when smooth); a local estimate while it catches up.
  const checked = validityMatches(path, validity);
  const local = checked ? null : localPathInfo(path);
  const length = checked ? validity.length_m : local!.length;
  const maxSpeed = checked ? validity.max_speed_mps : local!.maxSpeed;
  const n = path.waypoints.length;
  const reasons = validity?.reasons ?? [];
  const setOrientation = (o: Orientation) => {
    const target_id = o === 'target' ? (path.target_id && targets.some((x) => x.id === path.target_id) ? path.target_id : (targets[0]?.id ?? null)) : null;
    onPath({ ...path, orientation: o, target_id });
  };
  const along = smooth ? 'along the curve' : 'along the polyline';
  const summary = (
    <span
      className="small traj-summary"
      title={`Length ${fmtNum(length, 2)} m ${along} · from t = ${fmtNum(st.start, 2)} s to ${fmtNum(st.end, 2)} s · max speed ${fmtNum(maxSpeed, 2)} m/s${checked ? ' (as checked by the server)' : ' (estimated here: the check is being updated)'}`}
    >
      {n} waypoints{smooth ? ' · smooth' : ''} · {checked ? '' : '≈'}
      {fmtNum(length, 2)} m · {st.start > 0 ? `${fmtNum(st.start, 2)}–` : ''}
      {fmtNum(st.end, 2)} s · max {checked ? '' : '≈'}
      {fmtNum(maxSpeed, 2)} m/s
    </span>
  );

  return (
    <div className={`traj ${validity && !validity.valid ? 'invalid' : ''}`}>
      <div className="traj-head">
        <span className="traj-title">Trajectory</span>
        {summary}
        <div className="grow" />
        {expanded && (
          <button className="btn btn-xs btn-danger-ghost" disabled={locked} onClick={() => onPath(null)} title="Make the point static again (it stays at the first waypoint)">
            Remove
          </button>
        )}
      </div>
      {reasons.length > 0 && (
        <ul className="reasons">
          {reasons.map((r, j) => (
            <li key={j}>{r}</li>
          ))}
        </ul>
      )}
      {expanded && (
        <>
          <div className="point-fields">
            <label className="inline-field" title="Front axis of the point while it moves">
              <span>Facing</span>
              <select value={path.orientation} onChange={(e) => setOrientation(e.target.value as Orientation)} aria-label="Orientation along the trajectory">
                {ORIENTATIONS.map((o) => (
                  <option key={o.value} value={o.value} title={o.title} disabled={o.value === 'target' && targets.length === 0}>
                    {o.label}
                  </option>
                ))}
              </select>
            </label>
            {path.orientation === 'target' && (
              <select value={path.target_id ?? ''} onChange={(e) => onPath({ ...path, target_id: e.target.value || null })} aria-label="Target">
                {!targets.some((x) => x.id === path.target_id) && <option value={path.target_id ?? ''}>{path.target_id ?? '(none)'}</option>}
                {targets.map((x) => (
                  <option key={x.id} value={x.id}>
                    {x.name}
                  </option>
                ))}
              </select>
            )}
            {path.orientation === 'follow' && <span className="small muted">azimuth ignored</span>}
            <Toggle
              checked={smooth}
              onChange={(on) => onPath({ ...path, interpolation: on ? 'smooth' : 'linear' })}
              label="Smooth"
              disabled={locked}
              title={smooth ? SMOOTH_TITLE : `Straight lines between the waypoints. ${SMOOTH_TITLE}`}
            />
          </div>
          {smooth && <div className="small muted">Passes every waypoint at its time; the v column is the average speed between waypoints.</div>}
          <table className="table compact wp-table">
            <thead>
              <tr>
                <th>#</th>
                <th>X</th>
                <th>Y</th>
                <th>Z</th>
                <th title="Time at the waypoint (s)">t (s)</th>
                <th title={`${smooth ? 'Average speed (straight distance / time) of' : 'Speed of'} the segment arriving at the waypoint; editing it shifts the later times`}>v (m/s)</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {path.waypoints.map((w, k) => {
                const tb = timeBounds(path, k);
                const v = st.speeds[k];
                return (
                  <tr key={k} className={selectedWaypoint === k ? 'selected' : ''} onClick={() => onSelectWaypoint(k)}>
                    <td className="muted">{k + 1}</td>
                    {[0, 1, 2].map((a) => (
                      <td key={a}>
                        <NumField
                          value={w.position[a]}
                          step={0.1}
                          digits={3}
                          ariaLabel={`Waypoint ${k + 1} ${'XYZ'[a]} (m)`}
                          onChange={(nv) => {
                            const p: Vec3 = [...w.position];
                            p[a] = nv ?? 0;
                            onPath(setWaypointPosition(path, k, p));
                          }}
                        />
                      </td>
                    ))}
                    <td>
                      <NumField value={w.t} min={tb.min} max={tb.max} step={0.1} digits={3} ariaLabel={`Waypoint ${k + 1} time (s)`} onChange={(nv) => onPath(setWaypointTime(path, k, nv ?? w.t))} />
                    </td>
                    <td>
                      {k === 0 ? (
                        <span className="muted">–</span>
                      ) : v < 1e-9 ? (
                        <span className="muted" title="Same position as the previous waypoint: the point waits">
                          pause
                        </span>
                      ) : (
                        <NumField value={Number(v.toFixed(3))} min={0.01} max={340} step={0.1} digits={2} ariaLabel={`Speed into waypoint ${k + 1} (m/s)`} onChange={(nv) => nv && onPath(setSegmentSpeed(path, k, nv))} />
                      )}
                    </td>
                    <td className="wp-actions">
                      <button
                        className="icon-btn"
                        title={k < n - 1 ? 'Insert a waypoint halfway to the next one' : 'Add a waypoint 2 m further on'}
                        disabled={n >= MAX_WAYPOINTS}
                        onClick={(e) => {
                          e.stopPropagation();
                          onPath(insertWaypointAfter(path, k, item.azimuth_deg));
                          onSelectWaypoint(k + 1);
                        }}
                      >
                        +
                      </button>
                      <button
                        className="icon-btn"
                        title={n <= 2 ? 'A trajectory needs at least 2 waypoints' : 'Delete this waypoint'}
                        disabled={n <= 2}
                        onClick={(e) => {
                          e.stopPropagation();
                          onPath(deleteWaypoint(path, k));
                          onSelectWaypoint(null);
                        }}
                      >
                        ×
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div className="row gap">
            <button
              className={`btn btn-xs ${drawing ? 'btn-on' : ''}`}
              disabled={locked || n >= MAX_WAYPOINTS}
              onClick={(e) => {
                e.stopPropagation();
                onDraw(!drawing);
              }}
              title="Each click on a surface in the 3D view appends a waypoint (same height above the floor, 1.4 m/s)"
            >
              {drawing ? 'Stop adding' : 'Add waypoints'}
            </button>
            <span className="small muted">Positions in m. The point starts at waypoint 1.</span>
          </div>
        </>
      )}
    </div>
  );
}
