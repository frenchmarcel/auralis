import { useEffect, type MutableRefObject } from 'react';
import { useThree } from '@react-three/fiber';
import * as THREE from 'three';

export type CaptureBackground = 'theme' | 'white' | 'transparent' | 'custom';

export interface CaptureOptions {
  width: number;
  height: number;
  background: CaptureBackground;
  customColor: string;
  showSources: boolean;
  showReceivers: boolean;
  showLabels: boolean;
  showGrid: boolean;
  /** Transform gizmos (move / rotate handles) stay in the image. */
  showGizmos: boolean;
}

export const CAPTURE_MIN = 320;
export const CAPTURE_MAX = 4096;
export const THEME_BACKGROUND = '#111417';

/** Region of the viewport (CSS px) exported for an output of w x h: the largest such rectangle, centred, with a margin. */
export function captureFrame(viewW: number, viewH: number, w: number, h: number, margin = 24) {
  const aw = Math.max(1, viewW - 2 * margin);
  const ah = Math.max(1, viewH - 2 * margin);
  const scale = Math.min(aw / w, ah / h);
  const fw = w * scale;
  const fh = h * scale;
  return { x: (viewW - fw) / 2, y: (viewH - fh) / 2, w: fw, h: fh };
}

/** Marker groups carry `userData.capture = {kind, label, color, invalid}` (see Scene.tsx). */
interface CaptureTag {
  kind: 'source' | 'receiver';
  label: string;
  invalid?: boolean;
  group?: string | null;
}

export type CaptureFn = (o: CaptureOptions) => Promise<Blob>;

const LABEL_BG = { source: '#3ecf6e', receiver: '#4b8df0', invalid: '#e5484d' };

/**
 * Inside the Canvas: exposes `capture()` through `api`. It renders the frame region of
 * the current camera at the requested size into the drawing buffer, copies it to a 2D
 * canvas (labels are drawn there: the on-screen ones are HTML), then restores the view.
 */
export function CaptureBridge({ api }: { api: MutableRefObject<CaptureFn | null> }) {
  const gl = useThree((s) => s.gl);
  const scene = useThree((s) => s.scene);
  const camera = useThree((s) => s.camera);
  const size = useThree((s) => s.size);
  const invalidate = useThree((s) => s.invalidate);

  useEffect(() => {
    api.current = async (o) => {
      const W = Math.round(Math.min(CAPTURE_MAX, Math.max(CAPTURE_MIN, o.width)));
      const H = Math.round(Math.min(CAPTURE_MAX, Math.max(CAPTURE_MIN, o.height)));
      const fr = captureFrame(size.width, size.height, W, H);
      const ctx = gl.getContext();
      const maxRb = ctx.getParameter(ctx.MAX_RENDERBUFFER_SIZE) as number;
      const maxVp = (ctx.getParameter(ctx.MAX_VIEWPORT_DIMS) as Int32Array)[0];
      const limit = Math.min(maxRb, maxVp);
      if (W > limit || H > limit) throw new Error(`this graphics driver renders at most ${limit} px per side`);

      const hidden: THREE.Object3D[] = [];
      const labels: { tag: CaptureTag; pos: THREE.Vector3 }[] = [];
      scene.updateMatrixWorld(true);
      scene.traverse((obj) => {
        if (!shownInWorld(obj)) return;
        const tag = obj.userData?.capture as CaptureTag | undefined;
        const isGizmo = (obj as unknown as { isTransformControlsRoot?: boolean }).isTransformControlsRoot;
        let hide = false;
        if (tag) hide = tag.kind === 'source' ? !o.showSources : !o.showReceivers;
        else if (isGizmo) hide = !o.showGizmos;
        else if (obj.type === 'GridHelper') hide = !o.showGrid;
        if (hide) {
          hidden.push(obj);
          return;
        }
        if (tag && o.showLabels) labels.push({ tag, pos: obj.getWorldPosition(new THREE.Vector3()) });
      });
      hidden.forEach((obj) => (obj.visible = false));
      // Screen-sized points (px) keep their size relative to the image.
      const k = H / fr.h;
      const scaled: (() => void)[] = [];
      scene.traverse((obj) => {
        const m = (obj as THREE.Points).material as THREE.Material | undefined;
        if (!m || !(obj as THREE.Points).isPoints) return;
        if (m instanceof THREE.PointsMaterial && !m.sizeAttenuation) {
          const s0 = m.size;
          m.size = s0 * k;
          scaled.push(() => (m.size = s0));
        } else if (m instanceof THREE.ShaderMaterial && m.uniforms.uScale) {
          const u0 = m.uniforms.uScale.value as number;
          m.uniforms.uScale.value = (u0 * k) / gl.getPixelRatio();
          scaled.push(() => (m.uniforms.uScale.value = u0));
        }
      });

      const prevBg = scene.background;
      const prevClear = gl.getClearColor(new THREE.Color());
      const prevAlpha = gl.getClearAlpha();
      const prevDpr = gl.getPixelRatio();
      const prevAutoClear = gl.autoClear;
      const cam = camera as THREE.PerspectiveCamera | THREE.OrthographicCamera;
      const out = document.createElement('canvas');
      out.width = W;
      out.height = H;
      const c2 = out.getContext('2d');
      if (!c2) throw new Error('no 2D canvas');
      try {
        if (o.background === 'transparent') {
          scene.background = null;
          gl.setClearColor(0x000000, 0);
        } else {
          const bg = o.background === 'white' ? '#ffffff' : o.background === 'custom' ? o.customColor : THEME_BACKGROUND;
          scene.background = new THREE.Color(bg);
        }
        gl.setPixelRatio(1);
        gl.setSize(W, H, false);
        cam.setViewOffset(size.width, size.height, fr.x, fr.y, fr.w, fr.h);
        cam.updateProjectionMatrix();
        // A HUD (axes gizmo) leaves autoClear off: the background must be cleared here.
        gl.autoClear = true;
        gl.render(scene, cam);
        if (ctx.drawingBufferWidth < W || ctx.drawingBufferHeight < H) {
          throw new Error(`the browser limited the image to ${ctx.drawingBufferWidth} x ${ctx.drawingBufferHeight} px`);
        }
        // Same task as the render: the drawing buffer is still valid.
        c2.drawImage(gl.domElement, 0, 0, W, H);
        if (labels.length) drawLabels(c2, labels, cam, W, H, W / fr.w);
      } finally {
        hidden.forEach((obj) => (obj.visible = true));
        scaled.forEach((f) => f());
        gl.autoClear = prevAutoClear;
        scene.background = prevBg;
        gl.setClearColor(prevClear, prevAlpha);
        cam.clearViewOffset();
        cam.updateProjectionMatrix();
        gl.setPixelRatio(prevDpr);
        gl.setSize(size.width, size.height, false);
        invalidate();
      }
      return await new Promise<Blob>((resolve, reject) =>
        out.toBlob((b) => (b ? resolve(b) : reject(new Error('PNG encoding failed'))), 'image/png'),
      );
    };
    return () => {
      api.current = null;
    };
  }, [api, gl, scene, camera, size, invalidate]);
  return null;
}

function shownInWorld(o: THREE.Object3D): boolean {
  for (let x: THREE.Object3D | null = o; x; x = x.parent) if (!x.visible) return false;
  return true;
}

function drawLabels(
  c: CanvasRenderingContext2D,
  labels: { tag: CaptureTag; pos: THREE.Vector3 }[],
  cam: THREE.Camera,
  W: number,
  H: number,
  scale: number,
) {
  const k = Math.max(1, scale);
  const font = 10.5 * k;
  c.font = `600 ${font}px system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif`;
  c.textBaseline = 'middle';
  for (const { tag, pos } of labels) {
    const v = pos.clone().project(cam);
    if (v.z > 1 || v.z < -1) continue;
    const x = ((v.x + 1) / 2) * W;
    const y = ((1 - v.y) / 2) * H - 16 * k;
    const tw = c.measureText(tag.label).width;
    const padX = 5 * k;
    const bw = tw + 2 * padX + (tag.group ? 4 * k : 0);
    const bh = font + 5 * k;
    const bx = x - bw / 2;
    const by = y - bh / 2;
    c.fillStyle = 'rgba(0,0,0,0.35)';
    roundRect(c, bx + k, by + k, bw, bh, 3 * k);
    c.fill();
    c.fillStyle = tag.invalid ? LABEL_BG.invalid : LABEL_BG[tag.kind];
    roundRect(c, bx, by, bw, bh, 3 * k);
    c.fill();
    if (tag.group) {
      c.fillStyle = tag.group;
      c.fillRect(bx, by, 4 * k, bh);
    }
    c.fillStyle = tag.kind === 'source' && !tag.invalid ? '#0b0d0f' : '#ffffff';
    c.fillText(tag.label, bx + padX + (tag.group ? 4 * k : 0), y + 0.5 * k);
  }
}

function roundRect(c: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  c.beginPath();
  c.moveTo(x + r, y);
  c.lineTo(x + w - r, y);
  c.quadraticCurveTo(x + w, y, x + w, y + r);
  c.lineTo(x + w, y + h - r);
  c.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  c.lineTo(x + r, y + h);
  c.quadraticCurveTo(x, y + h, x, y + h - r);
  c.lineTo(x, y + r);
  c.quadraticCurveTo(x, y, x + r, y);
  c.closePath();
}
