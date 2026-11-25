import { useMemo, useState } from 'react';
import { Html, Line } from '@react-three/drei';
import * as THREE from 'three';
import type { Vec3 } from '../api';
import { COLORS } from '../lib/colors';

const noRaycast = () => null;
const UP = new THREE.Vector3(0, 1, 0);

/** A player as the ensemble would add it (POST .../ensemble with preview: true). */
export interface PreviewPlayer {
  id: string;
  name: string;
  position: Vec3;
  valid: boolean;
  reason: string | null;
  /** Display name of its directivity pattern. */
  pattern: string;
}

/** Ensemble dialog: the conductor, the audience direction and the players (accent colour, unlike real sources), invalid ones in red. */
export function EnsembleStagePreview({
  center,
  facingDeg,
  players,
  highlight,
  radius,
}: {
  center: Vec3;
  facingDeg: number;
  players: PreviewPlayer[] | null;
  /** Player hovered in the dialog's list. */
  highlight: string | null;
  radius: number;
}) {
  const [hover, setHover] = useState<string | null>(null);
  const shown = hover ?? highlight;
  const shownPlayer = players?.find((p) => p.id === shown) ?? null;
  const arrow = useMemo(() => {
    const f = (facingDeg * Math.PI) / 180;
    const d = new THREE.Vector3(Math.cos(f), Math.sin(f), 0);
    const len = radius * 16;
    const z = center[2] + 0.02;
    const tip: Vec3 = [center[0] + d.x * len, center[1] + d.y * len, z];
    return { from: [center[0], center[1], z] as Vec3, tip, q: new THREE.Quaternion().setFromUnitVectors(UP, d) };
  }, [center, facingDeg, radius]);
  return (
    <group>
      <mesh position={arrow.from} raycast={noRaycast} renderOrder={8}>
        <ringGeometry args={[radius * 1.1, radius * 1.7, 32]} />
        <meshBasicMaterial color={COLORS.accent} transparent opacity={0.95} depthTest={false} depthWrite={false} side={THREE.DoubleSide} />
      </mesh>
      <Line points={[arrow.from, arrow.tip]} color={COLORS.accent} lineWidth={2} transparent opacity={0.9} depthTest={false} renderOrder={8} raycast={noRaycast} />
      <mesh position={arrow.tip} quaternion={arrow.q} raycast={noRaycast} renderOrder={8}>
        <coneGeometry args={[radius * 0.7, radius * 1.8, 16]} />
        <meshBasicMaterial color={COLORS.accent} transparent opacity={0.95} depthTest={false} />
      </mesh>
      <Html position={arrow.tip} center zIndexRange={[19, 0]} style={{ pointerEvents: 'none' }}>
        <div className="ens-tag">audience</div>
      </Html>
      {players?.map((p) => {
        const color = p.valid ? COLORS.accent : COLORS.invalid;
        const on = p.id === shown;
        return (
          <mesh
            key={p.id}
            position={p.position}
            renderOrder={10}
            userData={{ isMarker: true }}
            onPointerOver={(e) => {
              e.stopPropagation();
              setHover(p.id);
            }}
            onPointerOut={() => setHover((h) => (h === p.id ? null : h))}
          >
            <sphereGeometry args={[radius * (on ? 1.05 : 0.75), 18, 12]} />
            <meshBasicMaterial color={on ? '#ffffff' : color} transparent opacity={0.9} depthTest={false} />
          </mesh>
        );
      })}
      {shownPlayer && (
        <Html position={[shownPlayer.position[0], shownPlayer.position[1], shownPlayer.position[2] + radius * 2]} center zIndexRange={[21, 0]} style={{ pointerEvents: 'none' }}>
          <div className={`ens-tip ${shownPlayer.valid ? '' : 'invalid'}`}>
            <b>{shownPlayer.name}</b> · {shownPlayer.pattern}
            {!shownPlayer.valid && <div>{shownPlayer.reason ?? 'invalid position'}</div>}
          </div>
        </Html>
      )}
    </group>
  );
}
