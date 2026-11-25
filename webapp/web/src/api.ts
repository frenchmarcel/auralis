export const BANDS_HZ = [125, 250, 500, 1000, 2000, 4000, 8000] as const;
export type Vec3 = [number, number, number];
/** Seven octave values (125 Hz..8 kHz); null where the server cannot compute. */
export type BandValues = (number | null)[];

export interface Health {
  native: boolean;
  gpu: boolean;
  version: string;
}

export type SimulationStatus = 'draft' | 'queued' | 'running' | 'done' | 'failed' | 'cancelled';
export type JobStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled';

export interface SimulationSummary {
  id: string;
  geometry_id: string;
  name: string;
  created: string;
  status: SimulationStatus;
  last_run: string | null;
  results_stale: boolean;
}

export interface RecentSimulation extends SimulationSummary {
  project_name: string;
  space_name: string;
  geometry_name: string;
  duration_s: number | null;
}

export interface GeometrySummary {
  id: string;
  space_id: string;
  name: string;
  created: string;
  n_triangles: number;
  n_layers: number;
  watertight: boolean;
  simulations: SimulationSummary[];
}

export interface Space {
  id: string;
  project_id: string;
  name: string;
  description: string;
  created: string;
  geometries: GeometrySummary[];
}

export interface Project {
  id: string;
  name: string;
  description: string;
  created: string;
  spaces: Space[];
}

export type FindingKind = 'open_edge' | 'nonmanifold_edge' | 'short_edge' | 'degenerate_triangle' | 'overlapping_faces' | 'small_gap';
export type Severity = 'error' | 'warning' | 'info';

export interface Finding {
  kind: FindingKind;
  severity: Severity;
  message: string;
  segment: [Vec3, Vec3];
  /** Stable id, used by exclusions (absent on older geometries). */
  id?: string;
  layers?: string[];
  excluded?: boolean;
  excluded_by?: 'finding' | 'layer';
  length_m?: number;
  /** overlapping_faces */
  overlap?: 'duplicate' | 'partial';
  contact?: boolean;
  area_m2?: number;
  n_triangles?: number;
  centroid?: Vec3;
  triangles?: [Vec3, Vec3, Vec3][];
  /** small_gap */
  gap_type?: 'slit' | 'layer';
  gap_m?: number;
  gap_max_m?: number;
  leak?: boolean;
}

/** Threshold of short edges and gaps: lambda/4 at the transition, or the fixed fallback. */
export interface CheckDetail {
  rule: 'lambda/4' | 'fixed';
  threshold_m: number;
  transition_hz: number | null;
  wave_dx_m: number | null;
  source: string;
}

export interface CheckExclusions {
  findings: string[];
  layers: string[];
}

export interface GeometryCheck {
  watertight: boolean;
  counts: {
    open_edges: number;
    nonmanifold_edges: number;
    short_edges: number;
    degenerate_triangles: number;
    overlapping_faces?: number;
    small_gaps?: number;
  };
  /** Excluded findings per kind (counts stay the totals). */
  excluded?: Partial<Record<FindingKind, number>>;
  findings: Finding[];
  solvers: { geometrical: boolean; wave: boolean };
  closure?: { method: 'topology' | 'voxels'; dx_m?: number; closed?: boolean };
  detail?: CheckDetail;
  exclusions?: CheckExclusions & { unmatched: string[] };
  /** What the model itself blocks (exclusions never change it). */
  blocking?: string[];
  /** Checks that could not run (e.g. a model too dense for the threshold). */
  notes?: string[];
}

export type LayerKind = 'room' | 'object';

export interface LayerInfo {
  name: string;
  n_triangles: number;
  area_m2: number;
  /** "object" for layers named obj:<catalog_id>:<part>; absent on older geometries (= room). */
  kind?: LayerKind;
}

export interface PlacedObject {
  id: string;
  catalog_id: string;
  name: string;
  position: Vec3;
  rotation_deg: number;
  scale: Vec3;
}

export type ProxyKind = 'upholstered' | 'chair' | 'plane_top' | 'box' | 'bed' | 'vertical_plane' | 'none';

export interface CatalogObject {
  id: string;
  name: string;
  category: string;
  source: string;
  license: string;
  proxy: ProxyKind | string;
  acoustic: boolean;
  parts: Record<string, { material: string; scattering: number | null }>;
  /** [width, depth, height] in metres, object frame (x width, y depth, z up). */
  dims: Vec3;
  visual: boolean;
  visual_url?: string | null;
  thumbnail_url?: string | null;
  visual_offset: Vec3;
  visual_scale: number;
  note?: string;
}

export interface ObjectCatalog {
  license_note: string;
  objects: CatalogObject[];
}

/** GET /textures: CC0 texture sets and HDRIs of the realistic view (display only). */
export interface TextureSetInfo {
  id: string;
  name: string;
  credit: string;
  /** Scanned area of the texture (m): one repeat along u and v. */
  size_m: [number, number];
  maps: { diff: string; nor?: string; rough?: string };
  license: string;
  url: string;
}

export interface TextureLibrary {
  available: boolean;
  textures: Record<string, TextureSetInfo>;
  hdri: Record<string, { id: string; name: string; credit: string; url_file: string; license: string; url: string }>;
}

/** POST /objects/proxy: proxy triangles per object layer at a placement (nothing saved). */
export interface ObjectProxyResponse {
  objects: { id: string; layers: Record<string, [Vec3, Vec3, Vec3][]> }[];
}

export type Unit = 'm' | 'cm' | 'mm';
export type UpAxis = 'z' | 'y';

export interface Geometry {
  id: string;
  space_id: string;
  name: string;
  created: string;
  source_filename: string;
  unit_in: Unit;
  up_axis_in: UpAxis;
  n_triangles: number;
  bbox: { min: Vec3; max: Vec3 };
  volume_m3: number | null;
  surface_m2: number;
  layers: LayerInfo[];
  check: GeometryCheck;
  /** Findings and layers the user marked as known (PUT /geometries/{gid}/exclusions). */
  check_exclusions?: CheckExclusions;
  objects?: PlacedObject[];
  /** Display-only .glb overlay (API "Visual model"); absent = none. */
  visual?: VisualModelInfo | null;
}

/** Placement of the visual model: up-axis conversion, scale, rotation X then Y then Z, translation. */
export interface VisualModelTransform {
  position: Vec3;
  rotation_deg: Vec3;
  scale: number;
  up_axis: UpAxis;
}

export interface VisualModelInfo {
  filename: string;
  bytes: number;
  uploaded: string;
  transform: VisualModelTransform;
}

export interface MeshData {
  positions: number[];
  /** owners: object id of each triangle (object layers only, same order as indices). */
  layers: { name: string; indices: number[]; kind?: LayerKind; owners?: string[] }[];
}

/** One sub-surface of a layer (display only: materials stay assigned per layer). */
export interface Subsurface {
  label: string;
  /** g/o group of the source OBJ, when the stored triangles match it. */
  group: string | null;
  orientation: 'horizontal' | 'vertical' | 'sloped' | 'curved' | null;
  /** Object layers: the placed instance. */
  object_id?: string;
  /** The layer's collinear (zero-area) triangles, grouped. */
  degenerate?: boolean;
  area_m2: number;
  n_triangles: number;
  /** Indices into the layer's triangles, in the order of GET .../mesh `indices`. */
  triangles: number[];
}

export interface SubsurfacesResponse {
  method: string;
  angle_deg: number;
  groups_from_source: boolean;
  note: string;
  layers: { name: string; by: 'patches' | 'instances'; subsurfaces: Subsurface[] }[];
}

export interface MaterialLibrary {
  id: string;
  name: string;
  redistributable: boolean;
  license: string;
}

export interface PhysicalModel {
  type: string;
  [key: string]: unknown;
}

export interface Material {
  id: string;
  library: string;
  name: string;
  category: string;
  description: string;
  absorption: number[];
  absorption_63: number | null;
  scattering: number;
  source: string;
  license: string;
  clipped: boolean;
  physical: PhysicalModel | null;
  /** Copies of library materials: false when the original may not be redistributed. */
  redistributable?: boolean;
  /** Copies: id of the material it was duplicated from. */
  derived_from?: string;
}

export interface MaterialsResponse {
  bands_hz: number[];
  libraries: MaterialLibrary[];
  materials: Material[];
}

export interface NewMaterialBody {
  name: string;
  category: string;
  description: string;
  absorption: number[];
  scattering: number;
  /** Optional 63 Hz alpha: the wave band follows it below 125 Hz; the rays never use it. */
  absorption_63?: number | null;
}

/** Porous layer: flow resistivity `sigma`, or a mineral wool and its bulk density instead. */
export interface PorousBody {
  model: 'miki';
  sigma?: number;
  wool?: WoolKind;
  density_kg_m3?: number;
  thickness_mm: number;
  air_gap_mm: number;
}

export type WoolKind = 'glass' | 'rock';

/** POST /materials/wool-sigma: flow resistivity of a mineral wool from its density (Pa·s/m², kg/m³). */
export interface WoolSigma {
  sigma: number;
  /** 1-sigma band of the law against its data. */
  sigma_low: number;
  sigma_high: number;
  uncertainty_pct: number;
  /** Density outside the data the law was checked on. */
  extrapolated: boolean;
  range_kg_m3: [number, number];
  law: string;
  source: string;
  wool: WoolKind;
  density_kg_m3: number;
}

export interface PorousPreview {
  absorption: number[];
  sigma: number;
  /** From a wool only. */
  wool?: WoolSigma;
  /** From a wool only: alphas at sigma_low and at sigma_high. */
  absorption_band?: [number[], number[]];
}

export type PerforatedKind = 'perforated' | 'slotted' | 'mpp';

/** Perforated / slotted / micro-perforated sheet over a plenum (mm, %, Pa·s/m, Pa·s/m²). */
export interface PerforatedBody {
  kind: PerforatedKind;
  size_mm: number;
  t_mm: number;
  open_pct: number;
  plenum_mm: number;
  wool_mm?: number;
  wool_sigma?: number;
  fleece_rayl?: number;
}

export interface NewPerforatedMaterialBody extends PerforatedBody {
  name: string;
  category: string;
  description: string;
  scattering: number;
}

export interface NewPorousMaterialBody extends PorousBody {
  name: string;
  category: string;
  description: string;
  scattering: number;
}

/** GET /directivities: presets and imported tables; the engine model is axisymmetric about the source's front axis. */
export interface Directivity {
  id: string;
  name: string;
  /** Absent on older servers (= preset). odeon_local: read in place from the local Odeon archives (research licence). */
  kind?: 'preset' | 'imported' | 'odeon_local';
  license?: string | null;
  /** Directivity index per octave (dB, 125 Hz-8 kHz). */
  di_db?: number[];
  /** Energy vs the angle from the axis, dB re the axis: theta 0-180 deg, db [7 bands][angles]. */
  polar?: { theta_deg: number[]; db: number[][] };
  /** Imported only, when the file carries it: on-axis SPL at 1 m per octave (sets the source level). */
  on_axis_spl_1m_db?: number[] | null;
  source?: string | null;
  method?: string | null;
  notes?: string[] | null;
  /** Largest spread between directions at the same angle (dB); the engine uses their mean. */
  asymmetry_db?: number | null;
}

export interface LayerAssignment {
  material_id: string | null;
  /** One value (flat over the bands) or 7 values (125 Hz-8 kHz); null = material default. */
  scattering: number | number[] | null;
}

export type Orientation = 'fixed' | 'follow' | 'target';

export interface Waypoint {
  position: Vec3;
  /** Seconds, >= 0, strictly increasing along the path. */
  t: number;
}

/** linear = straight lines; smooth = C1 cubic through every waypoint at its time (Catmull-Rom in time). */
export type Interpolation = 'linear' | 'smooth';

/** Trajectory of a source or receiver: timed waypoints, joined by straight lines or a smooth curve. */
export interface Path {
  waypoints: Waypoint[];
  /** fixed = azimuth/elevation; follow = heading along the path; target = aims at target_id. */
  orientation: Orientation;
  target_id: string | null;
  /** Absent = linear. */
  interpolation?: Interpolation;
}

export interface Source {
  id: string;
  name: string;
  position: Vec3;
  directivity: string;
  azimuth_deg: number;
  elevation_deg: number;
  gain_db: number;
  /** Per-octave EQ, 7 gains in dB (125 Hz - 8 kHz, within ±24); absent = flat (the server drops an all-zero one). */
  eq_db?: number[];
  /** Trajectory (null/absent = static); the server sets `position` to its first waypoint. */
  path?: Path | null;
  /** Track played by this source (ensembles); the auralizer's default sound. */
  sound_id?: string | null;
  /** Layer that vibrates instead of a point (wave solver only; `position` anchors the room's air). Absent/null = point source. */
  surface_layer?: string | null;
  /** Title of the ensemble the source belongs to (one source per player). */
  ensemble?: string | null;
  /** User group (list, colour); stored by the server as an extra field. */
  group?: PointGroup | null;
  /** Unknown server fields are kept as-is when saving. */
  [extra: string]: unknown;
}

export interface EnsembleTrack {
  sound_id: string;
  name: string;
}

export interface Ensemble {
  id: string;
  title: string;
  n_tracks: number;
  duration_s: number;
  credit: string;
  license: string;
  tracks: EnsembleTrack[];
}

export interface EnsembleBody {
  ensemble_id: string;
  /** Conductor position, on the floor. */
  center: Vec3;
  /** Direction of the audience (azimuth, CCW from +X). */
  facing_deg: number;
  width_m: number;
  depth_m: number;
  replace_sources: boolean;
}

export interface EnsembleResult {
  simulation: Simulation;
  /** < 1 when the layout was shrunk to fit width x depth. */
  scale: number;
  n_sources: number;
}

/** A player as the ensemble would add it, with its check (in the air volume, >= 0.10 m from surfaces). */
export interface EnsemblePreviewSource extends Source {
  valid: boolean;
  reason: string | null;
}

/** POST /simulations/{id}/ensemble with preview: true (nothing saved). */
export interface EnsemblePreview {
  preview: true;
  sources: EnsemblePreviewSource[];
  scale: number;
  n_sources: number;
  n_invalid: number;
}

export interface RenderStrip {
  source_id: string;
  sound_id: string;
  gain_db: number;
  muted: boolean;
}

export interface RenderBody {
  receiver_id: string;
  master_db: number;
  strips: RenderStrip[];
  /** binaural (default) or ambisonic: a 9-channel WAV (2nd order, ACN/SN3D, world axes). */
  format?: RenderFormat;
}

export type RenderFormat = 'binaural' | 'ambisonic';

/** GET /ambisonic/decoder: binaural decoding FIRs, head looking along +x. */
export interface AmbisonicDecoder {
  order: number;
  fs: number;
  head?: string;
  n_taps: number;
  latency_samples: number;
  convention: string;
  /** [9][2][n_taps] */
  filters: number[][][];
}

/** A named, coloured group of sources or receivers (Ctrl+G in the lists). */
export interface PointGroup {
  id: string;
  name: string;
  color: string;
}

/** GET /simulations/{id}/estimate: rough run time from unit costs measured on this machine. */
export interface RunEstimate {
  pairs_s: number;
  wave_s: number;
  grids_s: number;
  total_s: number;
  transition_hz: number | null;
  gpu: boolean;
  note: string;
}

export interface RenderResult {
  url: string;
  credits_url: string;
  fs: number;
  duration_s: number;
  normalization_db: number;
  credits: string[];
  n_sources: number;
  receiver_id?: string;
  format?: RenderFormat;
}

/** POST /simulations/{id}/mixes: the render of POST .../render as a background task. */
export interface MixTask {
  id: string;
  simulation_id: string;
  status: JobStatus;
  progress: number;
  /** e.g. "queued", "track 3 / 15", "done". */
  stage: string;
  /** The synchronous render's response, once done. */
  result: RenderResult | null;
  error: string | null;
  created?: string;
  started?: string | null;
  finished?: string | null;
}

/** Classes and spaces of an IFC file, before its import (API "Import a geometry"). */
export interface IfcInventory {
  classes: Record<string, number>;
  spaces: { id: string; name: string }[];
  schema: string;
  default_exclude: string[];
}

export interface Receiver {
  id: string;
  name: string;
  position: Vec3;
  azimuth_deg: number;
  elevation_deg: number;
  path?: Path | null;
  group?: PointGroup | null;
}

export type SettingsMode = 'survey' | 'default' | 'advanced';
export type SolverChoice = 'both' | 'wave' | 'geometrical';

export interface SimulationSettings {
  mode: SettingsMode;
  transition_hz: number | null;
  ir_length_s: number | null;
  ism_order: number;
  n_rays: number;
  solvers: SolverChoice;
  temperature_c: number;
  humidity_percent: number;
  diffraction: boolean;
  conformal: boolean;
  /** Declared open scene (object in free field); only meaningful for a non-closed model. */
  open_scene?: boolean;
  /** STI only: speech level at 1 m (IEC 60268-16 male spectrum), 60 dB(A) normal / 66 dB(A) raised. */
  speech_effort?: SpeechEffort;
  /** STI only: "none", an NC curve "NC-15" … "NC-70", or 7 octave levels (dB, 125 Hz–8 kHz). */
  background_noise?: BackgroundNoise;
  /** Advanced only, with ir_length_s null: IR long enough for this decay (dB, 20-90) in every band. */
  stop_decay_db?: number | null;
  /** Rays per receiver-grid point (1000-2 000 000, default 50 000). */
  grid_rays?: number;
  /** Advanced only: surface-size scattering (Christensen & Rindel 2005), on by default. */
  rbs_scattering?: boolean;
  /** Listener head of the binaural rendering: a /heads id ("model" = spherical model). */
  head?: string;
}

/** GET /heads: listener heads of the binaural rendering. */
export interface HeadInfo {
  id: string;
  name: string;
  /** false when its data is not installed locally (see data_hrtf/README.md). */
  available: boolean;
  credit: string | null;
}

export type SpeechEffort = 'normal' | 'raised';
export type BackgroundNoise = string | number[];

export interface Simulation {
  id: string;
  geometry_id: string;
  name: string;
  description: string;
  created: string;
  status: SimulationStatus;
  last_run: string | null;
  results_stale: boolean;
  layers: Record<string, LayerAssignment>;
  sources: Source[];
  receivers: Receiver[];
  settings: SimulationSettings;
  /** Surface receivers (absent on older simulations = none). */
  receiver_grids?: ReceiverGrid[];
}

export type SimulationEditable = Pick<
  Simulation,
  'name' | 'description' | 'layers' | 'sources' | 'receivers' | 'settings'
> & { receiver_grids?: ReceiverGrid[] };

export type GridKind = 'surface' | 'plane';

/** Surface receivers: a square lattice on a layer (offset along its normal) or on a free rectangle. */
export interface ReceiverGrid {
  id: string;
  name: string;
  kind: GridKind;
  /** Lattice step, 0.1-20 m; each point stands for spacing² of area. */
  spacing_m: number;
  /** surface: the layer, and the offset along its normal, towards the air (0.01-10 m). */
  layer?: string;
  offset_m?: number;
  /** plane: centre, width along the azimuth direction, length across it, tilt about the width axis. */
  center?: Vec3;
  width_m?: number;
  length_m?: number;
  azimuth_deg?: number;
  tilt_deg?: number;
}

/** Preview of a grid before any run. */
export interface GridPoints {
  grid_id: string;
  points: Vec3[];
  valid: boolean[];
  /** Why a point is disabled, by point index. */
  reasons: Record<string, string>;
  area_m2: number[];
  n_points: number;
  n_valid: number;
}

export interface GridValidity {
  id: string;
  n_points: number | null;
  n_valid: number;
  error: string | null;
}

export interface GridRunInfo {
  grid_id: string;
  sources: { source_id: string; n_points: number; n_valid: number; n_rays: number; rcv_radius_m: number; duration_s: number }[];
}

/** Grid parameters of one source: per point, 7 band values, or one value (STI, SPL_dba); null = disabled/undefined. */
export interface GridResults {
  grid_id: string;
  source_id: string;
  points: Vec3[];
  valid: boolean[];
  area_m2: number[];
  bands_hz: number[];
  params: Record<string, ((number | null)[] | number | null)[]>;
}

export interface CombineSource {
  source_id: string;
  gain_db: number;
  delay_ms: number;
  polarity: 1 | -1;
}

export interface CombineBody {
  receiver_id: string;
  sources: CombineSource[];
}

/** Coherent sum of sources at one receiver (cached server-side until the next run). */
export interface CombineResult {
  key: string;
  receiver_id: string;
  sources: CombineSource[];
  fs: number;
  metrics: PairMetrics;
  curves_url: string;
  wav_url: string;
}

export interface WaveFailure {
  source_id: string;
  error: string;
}

export interface PointValidity {
  id: string;
  valid: boolean;
  reasons: string[];
}

/** A trajectory checked every 0.1 m along it (same rules as a static point). */
export interface PathValidity extends PointValidity {
  length_m: number;
  start_s: number;
  end_s: number;
  max_speed_mps: number;
  /** Smooth paths only: the true curve, sampled at most 0.10 m apart (waypoints included). */
  curve?: Vec3[];
}

export interface Validation {
  ok_to_run: boolean;
  /** Every reason that keeps ok_to_run false (human-readable). */
  blocking?: string[];
  unassigned_layers: string[];
  sources: PointValidity[];
  receivers: PointValidity[];
  rt_estimate: { sabine: BandValues; eyring: BandValues } | null;
  /** Why there is no estimate (open scene), or null. */
  rt_note?: string | null;
  volume_m3: number | null;
  surface_m2: number | null;
  suggested_transition_hz: number | null;
  wave_available: boolean;
  watertight?: boolean;
  /** Effective open scene: declared AND the model is not closed (or closed only at the voxel scale). */
  open_scene?: boolean;
  /** How closure was established: edges match (topology) or enclosed air at the voxel scale. */
  closure_method?: 'topology' | 'voxels';
  /** One entry per point with a trajectory; never blocks a normal run. */
  paths?: PathValidity[];
  /** Non-blocking warnings (show them before Run). */
  warnings?: string[];
  /** One entry per receiver grid; an error (too many points) or no valid point blocks the run. */
  grids?: GridValidity[];
}

export type JobKind = 'simulation' | 'motion' | 'modal' | 'field';

export interface Job {
  id: string;
  /** Absent on older servers (= simulation). */
  kind?: JobKind;
  /** Motion jobs only: the animated render this job produces. */
  render_id?: string;
  simulation_id: string;
  status: JobStatus;
  progress: number;
  stage: string;
  sources: { id: string; name: string; progress: number }[];
  started: string | null;
  finished: string | null;
  eta_s: number | null;
  error: string | null;
  log: string[];
}

/** Per-band metrics (7 octave values each). */
export type MetricKey =
  | 'T20'
  | 'T30'
  | 'EDT'
  | 'C50'
  | 'C80'
  | 'D50'
  | 'Ts_ms'
  | 'SPL_db'
  | 'G_db'
  | 'MTI'
  | 'STI_signal_db'
  | 'JLF'
  | 'LJ_db';

/** Single-number metrics of a pair. */
export type ScalarKey = 'STI' | 'SPL_dbz' | 'SPL_dba' | 'SPL_dbc';

/** Any value may be null or absent (older results, undefined metric). */
export type PairMetrics = Partial<Record<MetricKey, BandValues>> &
  Partial<Record<ScalarKey, number | null>> & {
    /** Decay reached by the end of the IR per band (dB); above -45 dB T30 is unreliable. */
    decay_range_db?: BandValues;
    /** 63 Hz octave (44-88 Hz) from the wave band alone; only when the wave solver ran with a transition >= 125 Hz. */
    band63?: Band63 | null;
    /** Same parameters on the geometrical solver alone and on the wave band alone (wave-solver runs, v0.6+). */
    by_solver?: BySolver | null;
  };

/** Parameters the server splits by solver ("result type"). */
export type SolverMetricKey = 'T20' | 'T30' | 'EDT' | 'C50' | 'C80' | 'D50' | 'Ts_ms';
/** geo: full band, geometrical solver alone; wave: wave band alone (bands above the transition are null). */
export type BySolver = Partial<Record<'geo' | 'wave', Partial<Record<SolverMetricKey, BandValues>>>>;
/** Treble's "result type": hybrid (default), geometrical solver alone, wave band alone. */
export type ResultPart = 'hybrid' | 'geo' | 'wave';

/** Single values of the 63 Hz octave (wave band only); no G/SPL. */
export type Band63 = Partial<Record<'T20' | 'T30' | 'EDT' | 'C50' | 'C80' | 'D50' | 'Ts_ms' | 'G_db' | 'SPL_db', number | null>>;

export interface ResultPair {
  source_id: string;
  receiver_id: string;
  metrics: PairMetrics;
  timings: { total_s?: number; [key: string]: number | undefined };
}

export interface Results {
  bands_hz: number[];
  fs: number;
  transition_hz: number | null;
  engine_used: string;
  ir_scale: number;
  stale: boolean;
  duration_s?: number;
  settings: SimulationSettings;
  pairs: ResultPair[];
  /** "omni": ISO 3382 parameters from the omnidirectional IR; "binaural mean": runs before v0.5. */
  metrics_ir?: 'omni' | 'binaural mean';
  /** Sources whose wave band failed: they use the geometrical solver over the whole band. */
  wave_failed?: WaveFailure[];
  grids?: GridRunInfo[];
}

export type FrSmoothing = 0 | 24 | 12 | 6 | 3 | 1;

export interface Curves {
  fs: number;
  /** `omni` when the run produced it; for part=geo|wave, left = right = omni. */
  ir: { t: number[]; left: number[]; right: number[]; omni?: number[] };
  edc: Record<string, { t: number[]; db: number[] }>;
  fr: { f: number[]; db: number[] };
  /** Which result the curves are of (absent on older servers = hybrid). */
  part?: ResultPart;
  source?: string;
}

export type ArrivalKind = 'direct' | 'specular' | 'diffracted';

/** One deterministic arrival (reflection tracking). */
export interface Arrival {
  kind: ArrivalKind;
  order: number;
  time_ms: number;
  delay_ms: number;
  distance_m: number;
  /** Per octave band, dB re 20 µPa for a source of 94 dB at 1 m (plus gain and directivity). */
  spl_db: BandValues;
  level_db: number | null;
  /** Relative to the direct sound; null when the direct sound is occluded. */
  level_rel_db: number | null;
  /** Direction of arrival, world frame (CCW from +X). */
  azimuth_deg: number;
  /** Relative to the receiver's look direction (+ = to its left). */
  azimuth_rel_deg: number;
  elevation_deg: number;
  /** Layer of each reflection, source side first. */
  surfaces: string[];
  /** Source, reflection/diffraction points in order, receiver. */
  path: Vec3[];
}

export interface Reflections {
  source_id: string;
  receiver_id: string;
  max_order: number;
  speed_of_sound: number;
  spl_1m_db: number;
  n_direct_occluded: number;
  events: Arrival[];
}

export interface MotionCheckBody {
  receiver_id: string;
  /** null = until the last waypoint of the moving points involved. */
  duration_s: number | null;
  strips: RenderStrip[];
}

export interface MotionRenderBody extends MotionCheckBody {
  doppler: boolean;
  /** Late-field key frame interval (s). */
  reverb_dt: number;
  master_db: number;
  label: string;
  /** Conversation: the strips' recordings speak in turn, cut at their pauses (null = all at once). */
  turn_taking?: { gap_s: number } | null;
  /** Mix the simulation's background noise; the tracks are calibrated as speech at the vocal effort. */
  background_noise?: boolean;
}

export interface MotionCheck {
  ok: boolean;
  blocking: string[];
  warnings: string[];
  duration_s: number;
  path_end_s: number;
  paths: PathValidity[];
}

export interface MotionRenderResult {
  fs: number;
  duration_s: number;
  render_s: number;
  normalization_db: number;
  credits: string[];
  warnings: string[];
  doppler: boolean;
  reverb_dt?: number;
  turns?: { source_id: string; start_s: number; end_s: number }[] | null;
  background_noise?: { curve: string; dba: number; speech_dba_1m: number; effort: string } | null;
  [extra: string]: unknown;
}

export interface MotionRender {
  id: string;
  simulation_id: string;
  created: string;
  status: JobStatus;
  receiver_id: string;
  label: string;
  n_sources: number;
  error: string | null;
  job_id?: string | null;
  finished?: string | null;
  /** Only once done. */
  url?: string;
  credits_url?: string;
  result?: MotionRenderResult;
}

export interface Sound {
  id: string;
  name: string;
  /** generated | uploaded | odeon_local (local research licence, not redistributable) */
  kind: string;
  duration_s: number;
  /** Credit notice that must accompany any result using this recording. */
  credit?: string;
  license?: string;
  /** odeon_local only: concerts | misc */
  group?: string;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function detailToMessage(detail: unknown): string | null {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    // FastAPI validation errors: [{loc, msg, type}, ...]
    return detail
      .map((d) => {
        if (d && typeof d === 'object' && 'msg' in d) {
          const loc = Array.isArray((d as { loc?: unknown[] }).loc)
            ? (d as { loc: unknown[] }).loc.join('.')
            : '';
          return loc ? `${loc}: ${(d as { msg: string }).msg}` : (d as { msg: string }).msg;
        }
        return JSON.stringify(d);
      })
      .join('; ');
  }
  if (detail != null) return JSON.stringify(detail);
  return null;
}

async function parseError(res: Response): Promise<ApiError> {
  let message = `${res.status} ${res.statusText}`;
  try {
    const text = await res.text();
    try {
      const body = JSON.parse(text);
      const m = detailToMessage(body?.detail);
      if (m) message = m;
    } catch {
      if (text) message = `${message}: ${text.slice(0, 300)}`;
    }
  } catch {
    /* body unreadable: keep the status line */
  }
  return new ApiError(res.status, message);
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/api${path}`, {
      method,
      headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch (e) {
    throw new ApiError(0, `Server unreachable (${(e as Error).message})`);
  }
  if (!res.ok) throw await parseError(res);
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

/** Multipart upload through XHR so that upload progress can be reported. */
function upload<T>(path: string, form: FormData, onProgress?: (frac: number) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `/api${path}`);
    xhr.responseType = 'text';
    if (onProgress) {
      xhr.upload.onprogress = (ev) => {
        if (ev.lengthComputable) onProgress(ev.loaded / ev.total);
      };
    }
    xhr.onerror = () => reject(new ApiError(0, 'Server unreachable during upload'));
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as T);
        } catch {
          reject(new ApiError(xhr.status, 'Invalid JSON in server response'));
        }
        return;
      }
      let message = `${xhr.status} ${xhr.statusText}`;
      try {
        const m = detailToMessage(JSON.parse(xhr.responseText)?.detail);
        if (m) message = m;
      } catch {
        if (xhr.responseText) message = `${message}: ${xhr.responseText.slice(0, 300)}`;
      }
      reject(new ApiError(xhr.status, message));
    };
    xhr.send(form);
  });
}

const enc = encodeURIComponent;

export const api = {
  health: () => request<Health>('GET', '/health'),

  listProjects: () => request<Project[]>('GET', '/projects'),
  recentSimulations: () => request<RecentSimulation[]>('GET', '/recent-simulations'),
  createProject: (name: string, description = '') =>
    request<Project>('POST', '/projects', { name, description }),
  updateProject: (pid: string, patch: { name?: string; description?: string }) =>
    request<Project>('PATCH', `/projects/${enc(pid)}`, patch),
  deleteProject: (pid: string) => request<{ ok: boolean }>('DELETE', `/projects/${enc(pid)}`),

  createSpace: (pid: string, name: string, description = '') =>
    request<Space>('POST', `/projects/${enc(pid)}/spaces`, { name, description }),
  updateSpace: (sid: string, patch: { name?: string; description?: string }) =>
    request<Space>('PATCH', `/spaces/${enc(sid)}`, patch),
  deleteSpace: (sid: string) => request<{ ok: boolean }>('DELETE', `/spaces/${enc(sid)}`),

  importGeometry: (
    sid: string,
    args: { file: File; name: string; up_axis: UpAxis; unit: Unit; ifc_space?: string; ifc_exclude?: string[] },
    onProgress?: (frac: number) => void,
  ) => {
    const form = new FormData();
    form.append('file', args.file, args.file.name);
    form.append('name', args.name);
    form.append('up_axis', args.up_axis);
    form.append('unit', args.unit);
    if (args.ifc_space !== undefined) form.append('ifc_space', args.ifc_space);
    if (args.ifc_exclude !== undefined) form.append('ifc_exclude', args.ifc_exclude.join(','));
    return upload<Geometry>(`/spaces/${enc(sid)}/geometries`, form, onProgress);
  },
  ifcInventory: (file: File) => {
    const form = new FormData();
    form.append('file', file, file.name);
    return upload<IfcInventory>('/import/ifc-inventory', form);
  },
  getGeometry: (gid: string) => request<Geometry>('GET', `/geometries/${enc(gid)}`),
  updateGeometry: (gid: string, patch: { name?: string }) =>
    request<Geometry>('PATCH', `/geometries/${enc(gid)}`, patch),
  deleteGeometry: (gid: string) => request<{ ok: boolean }>('DELETE', `/geometries/${enc(gid)}`),
  getMesh: (gid: string) => request<MeshData>('GET', `/geometries/${enc(gid)}/mesh`),
  /** Re-check: lambda/4 at transition_hz, else at the simulation's transition, else fixed. */
  checkGeometry: (gid: string, body: { transition_hz?: number | null; simulation_id?: string | null }) =>
    request<Geometry>('POST', `/geometries/${enc(gid)}/check`, body),
  putExclusions: (gid: string, ex: CheckExclusions) => request<Geometry>('PUT', `/geometries/${enc(gid)}/exclusions`, ex),
  getSubsurfaces: (gid: string) => request<SubsurfacesResponse>('GET', `/geometries/${enc(gid)}/subsurfaces`),
  /** The imported model file, as a blob with the name the server gives it. */
  fetchSourceModel: async (gid: string): Promise<{ blob: Blob; filename: string }> => {
    let res: Response;
    try {
      res = await fetch(`/api/geometries/${enc(gid)}/source`);
    } catch (e) {
      throw new ApiError(0, `Server unreachable (${(e as Error).message})`);
    }
    if (!res.ok) throw await parseError(res);
    const cd = res.headers.get('Content-Disposition') ?? '';
    const star = /filename\*=utf-8''([^;]+)/i.exec(cd);
    const plain = /filename="([^"]*)"/i.exec(cd);
    const filename = star ? decodeURIComponent(star[1]) : plain ? plain[1] : 'source';
    return { blob: await res.blob(), filename };
  },
  duplicateGeometry: (gid: string, name: string) => request<Geometry>('POST', `/geometries/${enc(gid)}/duplicate`, { name }),
  putObjects: (gid: string, objects: PlacedObject[]) => request<Geometry>('PUT', `/geometries/${enc(gid)}/objects`, { objects }),
  getObjectCatalog: () => request<ObjectCatalog>('GET', '/objects/catalog'),
  getTextures: () => request<TextureLibrary>('GET', '/textures'),
  objectProxy: (objects: Omit<PlacedObject, 'name'>[]) => request<ObjectProxyResponse>('POST', '/objects/proxy', { objects }),
  uploadVisual: (gid: string, file: File, onProgress?: (frac: number) => void) => {
    const form = new FormData();
    form.append('file', file, file.name);
    return upload<Geometry>(`/geometries/${enc(gid)}/visual`, form, onProgress);
  },
  /** The uploaded file; `uploaded` busts the browser cache after a replacement. */
  visualUrl: (gid: string, uploaded?: string) => `/api/geometries/${enc(gid)}/visual.glb${uploaded ? `?v=${enc(uploaded)}` : ''}`,
  setVisualTransform: (gid: string, patch: Partial<VisualModelTransform>) =>
    request<Geometry>('PUT', `/geometries/${enc(gid)}/visual`, patch),
  /** Only for a server that returns `visual.lighting` (see lib/visualLighting). */
  setVisualLighting: (gid: string, lighting: object | null) => request<Geometry>('PUT', `/geometries/${enc(gid)}/visual`, { lighting }),
  deleteVisual:(gid: string) => request<Geometry>('DELETE', `/geometries/${enc(gid)}/visual`),

  getMaterials: () => request<MaterialsResponse>('GET', '/materials'),
  createMaterial: (body: NewMaterialBody) => request<Material>('POST', '/materials', body),
  woolSigma: (body: { wool: WoolKind; density_kg_m3: number }) => request<WoolSigma>('POST', '/materials/wool-sigma', body),
  porousPreview: (body: PorousBody) =>
    request<PorousPreview>('POST', '/materials/porous-preview', body),
  createPorousMaterial: (body: NewPorousMaterialBody) =>
    request<Material>('POST', '/materials/porous', body),
  perforatedPreview: (body: PerforatedBody) =>
    request<{ absorption: number[] }>('POST', '/materials/perforated-preview', body),
  createPerforatedMaterial: (body: NewPerforatedMaterialBody) =>
    request<Material>('POST', '/materials/perforated', body),
  deleteMaterial: (id: string) => request<{ ok: boolean }>('DELETE', `/materials/${enc(id)}`),
  /** Custom materials only; the id is kept, so simulations keep their assignment. */
  renameMaterial: (id: string, name: string) => request<Material>('PATCH', `/materials/${enc(id)}`, { name }),
  /** Any material, copied into a new custom one (licence kept); no name = "<name> (copy)". */
  duplicateMaterial: (id: string, name?: string) =>
    request<Material>('POST', `/materials/${enc(id)}/duplicate`, name ? { name } : {}),

  getDirectivities: () => request<Directivity[]>('GET', '/directivities'),

  createSimulation: (gid: string, name: string, copy_from: string | null = null) =>
    request<Simulation>('POST', `/geometries/${enc(gid)}/simulations`, { name, copy_from }),
  getSimulation: (id: string) => request<Simulation>('GET', `/simulations/${enc(id)}`),
  saveSimulation: (id: string, body: SimulationEditable) =>
    request<Simulation>('PUT', `/simulations/${enc(id)}`, body),
  deleteSimulation: (id: string) => request<{ ok: boolean }>('DELETE', `/simulations/${enc(id)}`),
  getValidation: (id: string) => request<Validation>('GET', `/simulations/${enc(id)}/validation`),

  runSimulation: (id: string) => request<Job>('POST', `/simulations/${enc(id)}/run`),
  getJob: (jid: string) => request<Job>('GET', `/jobs/${enc(jid)}`),
  listJobs: () => request<Job[]>('GET', '/jobs'),
  cancelJob: (jid: string) => request<Job>('POST', `/jobs/${enc(jid)}/cancel`),

  getResults: (id: string) => request<Results>('GET', `/simulations/${enc(id)}/results`),
  getCurves: (id: string, sourceId: string, receiverId: string, frSmoothing: FrSmoothing, part: ResultPart = 'hybrid') =>
    request<Curves>(
      'GET',
      `/simulations/${enc(id)}/results/${enc(sourceId)}/${enc(receiverId)}/curves?fr_smoothing=${frSmoothing}${part !== 'hybrid' ? `&part=${part}` : ''}`,
    ),
  irWavUrl: (id: string, sourceId: string, receiverId: string) =>
    `/api/simulations/${enc(id)}/results/${enc(sourceId)}/${enc(receiverId)}/ir.wav`,
  exportXlsxUrl: (id: string) => `/api/simulations/${enc(id)}/export.xlsx`,
  irsZipUrl: (id: string) => `/api/simulations/${enc(id)}/irs.zip`,
  getGridPoints: (id: string, gridId: string) => request<GridPoints>('GET', `/simulations/${enc(id)}/grids/${enc(gridId)}/points`),
  getGridResults: (id: string, gridId: string, sourceId: string, param?: string) =>
    request<GridResults>(
      'GET',
      `/simulations/${enc(id)}/grids/${enc(gridId)}/results?source_id=${enc(sourceId)}${param ? `&param=${enc(param)}` : ''}`,
    ),
  combineSources: (id: string, body: CombineBody) => request<CombineResult>('POST', `/simulations/${enc(id)}/combine`, body),
  /** `curvesUrl` as returned by combineSources (starts with /api). */
  getCombinedCurves: (curvesUrl: string, frSmoothing: FrSmoothing) =>
    request<Curves>('GET', `${curvesUrl.replace(/^\/api/, '')}?fr_smoothing=${frSmoothing}`),
  /** `look` overrides the receiver's look direction for azimuth_rel_deg (the listener turns the head). */
  getReflections: (id: string, sourceId: string, receiverId: string, maxOrder: number, look?: { azimuth_deg: number; elevation_deg: number } | null) =>
    request<Reflections>(
      'GET',
      `/simulations/${enc(id)}/reflections?source_id=${enc(sourceId)}&receiver_id=${enc(receiverId)}&max_order=${maxOrder}${look ? `&azimuth_deg=${look.azimuth_deg}&elevation_deg=${look.elevation_deg}` : ''}`,
    ),

  listEnsembles: () => request<Ensemble[]>('GET', '/ensembles'),
  addEnsemble: (simId: string, body: EnsembleBody) => request<EnsembleResult>('POST', `/simulations/${enc(simId)}/ensemble`, body),
  previewEnsemble: (simId: string, body: EnsembleBody) =>
    request<EnsemblePreview>('POST', `/simulations/${enc(simId)}/ensemble`, { ...body, preview: true }),
  renderMix: (simId: string, body: RenderBody) => request<RenderResult>('POST', `/simulations/${enc(simId)}/render`, body),
  startMix: (simId: string, body: RenderBody) => request<MixTask>('POST', `/simulations/${enc(simId)}/mixes`, body),
  getMix: (simId: string, rid: string) => request<MixTask>('GET', `/simulations/${enc(simId)}/mixes/${enc(rid)}`),
  cancelMix: (simId: string, rid: string) => request<MixTask>('POST', `/simulations/${enc(simId)}/mixes/${enc(rid)}/cancel`),
  getAmbisonicDecoder: (fs: number, head = 'model') =>
    request<AmbisonicDecoder>('GET', `/ambisonic/decoder?fs=${fs}&head=${enc(head)}`),
  getHeads: () => request<HeadInfo[]>('GET', '/heads'),
  getEstimate: (id: string) => request<RunEstimate>('GET', `/simulations/${enc(id)}/estimate`),

  checkMotion: (simId: string, body: MotionCheckBody) => request<MotionCheck>('POST', `/simulations/${enc(simId)}/motion/check`, body),
  submitMotion: (simId: string, body: MotionRenderBody) =>
    request<{ render: MotionRender; job: Job }>('POST', `/simulations/${enc(simId)}/motion`, body),
  listMotion: (simId: string) => request<MotionRender[]>('GET', `/simulations/${enc(simId)}/motion`),
  getMotion: (simId: string, rid: string) => request<MotionRender>('GET', `/simulations/${enc(simId)}/motion/${enc(rid)}`),
  deleteMotion: (simId: string, rid: string) => request<{ ok: boolean }>('DELETE', `/simulations/${enc(simId)}/motion/${enc(rid)}`),

  listSounds: () => request<Sound[]>('GET', '/sounds'),
  soundWavUrl: (id: string) => `/api/sounds/${enc(id)}.wav`,
  uploadSound: (file: File, name: string) => {
    const form = new FormData();
    form.append('file', file, file.name);
    form.append('name', name);
    return upload<Sound>('/sounds', form);
  },

  getMaterialDetails: (id: string) => request<MaterialDetails>('GET', `/materials/${enc(id)}/details`),
  createTableMaterial: (body: TableMaterialBody) => request<Material>('POST', '/materials/table', body),

  importDirectivity: (file: File, name: string) => {
    const form = new FormData();
    form.append('file', file, file.name);
    form.append('name', name);
    return upload<Directivity>('/directivities', form);
  },
  deleteDirectivity: (id: string) => request<{ ok: boolean }>('DELETE', `/directivities/${enc(id)}`),

  getOffice: (simId: string, sourceId: string, receiverIds: string[]) =>
    request<OfficeResult>('GET', `/simulations/${enc(simId)}/iso3382-3?source_id=${enc(sourceId)}&receiver_ids=${receiverIds.map(enc).join(',')}`),

  listModal: (simId: string) => request<ModalTask[]>('GET', `/simulations/${enc(simId)}/modal`),
  getModal: (simId: string, rid: string) => request<ModalTask>('GET', `/simulations/${enc(simId)}/modal/${enc(rid)}`),
  submitModal: (simId: string, body: ModalBody) => request<{ modal: ModalTask; job: Job }>('POST', `/simulations/${enc(simId)}/modal`, body),
  /** One frequency (the nearest computed) or, without `freq`, all of them. */
  getModalMap: (simId: string, rid: string, freq?: number) =>
    request<ModalMap>('GET', `/simulations/${enc(simId)}/modal/${enc(rid)}/map${freq !== undefined ? `?freq=${freq}` : ''}`),
  deleteModal: (simId: string, rid: string) => request<{ ok: boolean }>('DELETE', `/simulations/${enc(simId)}/modal/${enc(rid)}`),

  listField: (simId: string) => request<FieldTask[]>('GET', `/simulations/${enc(simId)}/field`),
  getField: (simId: string, rid: string) => request<FieldTask>('GET', `/simulations/${enc(simId)}/field/${enc(rid)}`),
  submitField: (simId: string, body: FieldBody) => request<{ field: FieldTask; job: Job }>('POST', `/simulations/${enc(simId)}/field`, body),
  getFieldFrames: (simId: string, rid: string) => request<FieldFrames>('GET', `/simulations/${enc(simId)}/field/${enc(rid)}/frames`),
  deleteField: (simId: string, rid: string) => request<{ ok: boolean }>('DELETE', `/simulations/${enc(simId)}/field/${enc(rid)}`),

  getUiState: (simId: string, key: UiStateKey) => request<Record<string, unknown>>('GET', `/simulations/${enc(simId)}/ui-state/${key}`),
  putUiState: (simId: string, key: UiStateKey, body: Record<string, unknown>) =>
    request<Record<string, unknown>>('PUT', `/simulations/${enc(simId)}/ui-state/${key}`, body),
  /** Same PUT, kept alive while the page unloads. */
  putUiStateKeepalive: (simId: string, key: UiStateKey, body: Record<string, unknown>) =>
    fetch(`/api/simulations/${enc(simId)}/ui-state/${key}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), keepalive: true }),
};

/** GET /materials/{id}/details: what the engine uses for this material. */
export interface MaterialDetails {
  id: string;
  bands_hz: number[];
  alpha_table: number[];
  /** The material's 63 Hz value, or null (the wall then holds its 125 Hz behaviour). */
  alpha_table_63: number | null;
  /** Admittance fit BEFORE the per-band anchoring that makes rays and reflections match alpha_table. */
  alpha_fit_random: number[];
  /** The fit at 63 Hz: what the wave band sees there (no anchoring below 125 Hz). */
  alpha_fit_63: number;
  /** 64 log-spaced frequencies, 50 Hz-10 kHz. */
  freqs_hz: number[];
  R_normal: { re: number[]; im: number[]; mag: number[]; phase_deg: number[] };
  alpha_normal: number[];
  /** Whether the phase is known (porous model, measured table) or a model assumption (alpha-only fit). */
  phase_source: string;
  /** For an alpha-only material: the physical model inferred for the wave band, or why none was. */
  wave_model: string | null;
}

/** z = surface impedance / (rho0 c0); r = reflection factor (normal incidence). */
export type TableKind = 'z' | 'r';

export interface TableMaterialBody {
  name: string;
  category?: string;
  description?: string;
  kind: TableKind;
  freqs: number[];
  re: number[];
  im: number[];
  scattering: number;
}

/** ISO 3382-3:2022 along a path of receivers (GET /iso3382-3). */
export interface OfficePosition {
  receiver_id: string;
  distance_m: number;
  Lp_A_S_db: number;
  STI: number;
  Lp_S_db: number[];
}

export interface OfficeResult {
  positions: OfficePosition[];
  /** Positions 2-16 m from the source, used by the regression. */
  n_in_range: number;
  D2S_db: number | null;
  Lp_A_S_4m_db: number | null;
  rC_m: number | null;
  rC_note: string | null;
  rD_m: number | null;
  rD_note: string | null;
  Lp_A_B_db: number | null;
  directivity_note: string | null;
}

export interface ModalBody {
  source_id: string;
  grid_id: string;
  f_min?: number;
  f_max?: number;
  step_hz?: number;
  freqs?: number[];
  label?: string;
}

export interface ModalResultMeta {
  source_id: string;
  grid_id: string;
  n_points: number;
  n_valid: number;
  /** Frequencies actually used (nearest FFT bins). */
  freqs_hz: number[];
  dx_m: number;
  duration_s: number;
  engine: string;
  conformal: boolean;
  render_s: number;
  note: string;
  [extra: string]: unknown;
}

/** A forced-response (modal) map task. */
export interface ModalTask {
  id: string;
  simulation_id: string;
  kind: 'modal';
  created: string;
  status: JobStatus;
  label: string;
  error: string | null;
  source_id: string;
  grid_id: string;
  n_freqs: number;
  job_id?: string | null;
  finished?: string | null;
  /** Only once done. */
  result?: ModalResultMeta;
}

/** Level in dB re the same source in free field at 10 m (G convention), [frequency][point]. */
export interface ModalMap {
  points: Vec3[];
  valid: boolean[];
  area_m2: number[];
  freqs_hz: number[];
  level_db: (number | null)[][];
}

export interface FieldBody {
  source_id: string;
  grid_id: string;
  /** 50-1000 Hz. */
  f_max_hz: number;
  /** 0.02-1 s. */
  duration_s: number;
  /** >= 2.5 x f_max; duration x fps <= 2000. */
  fps: number;
  label?: string;
}

export interface FieldResultMeta {
  source_id: string;
  grid_id: string;
  n_points: number;
  n_valid: number;
  fps: number;
  n_frames: number;
  duration_s: number;
  f_max_hz: number;
  dx_m: number;
  engine: string;
  conformal?: boolean;
  render_s?: number;
  note: string;
  [extra: string]: unknown;
}

/** A wave-field animation task. */
export interface FieldTask {
  id: string;
  simulation_id: string;
  kind: 'field';
  created: string;
  status: JobStatus;
  label: string;
  error: string | null;
  source_id: string;
  grid_id: string;
  job_id?: string | null;
  finished?: string | null;
  /** Only once done. */
  result?: FieldResultMeta;
}

/** int8 frames x points (row-major, base64): value = byte / 127 x scale. */
export interface FieldFrames {
  points: Vec3[];
  valid: boolean[];
  /** Into the air. */
  normals: Vec3[];
  /** First lattice axis of each point. */
  u_axes: Vec3[];
  n_frames: number;
  n_points: number;
  scale: number;
  data_b64: string;
}

export type UiStateKey = 'results' | 'auralizer' | 'editor';

export function errorMessage(e: unknown): string {
  if (e instanceof Error) return e.message;
  return String(e);
}

/** Pick the editable subset of a simulation for PUT. */
export function editableOf(sim: Simulation): SimulationEditable {
  return {
    name: sim.name,
    description: sim.description,
    layers: sim.layers,
    sources: sim.sources,
    receivers: sim.receivers,
    settings: sim.settings,
    ...(sim.receiver_grids ? { receiver_grids: sim.receiver_grids } : {}),
  };
}
