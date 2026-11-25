import type { Geometry, MeshData } from '../../api';
import { Modal } from '../../components/ui';
import { ReviewStep } from '../ImportWizard';

/** The geometry checker of the editor: re-checks against this simulation's transition when opened. */
export function GeometryCheckDialog({
  geometry,
  mesh,
  simulationId,
  onGeometry,
  onClose,
}: {
  geometry: Geometry;
  mesh: MeshData;
  simulationId: string;
  onGeometry: (g: Geometry) => void;
  onClose: () => void;
}) {
  return (
    <Modal title={<>Geometry check · {geometry.name}</>} onClose={onClose} width="94vw" className="modal-tall">
      <ReviewStep geometry={geometry} mesh={mesh} simulationId={simulationId} autoCheck onGeometry={onGeometry} />
      <div className="modal-footer">
        <span className="muted small">Uses the saved settings of this simulation. Exclusions only mark findings as known: the solvers and the run checks are unchanged.</span>
        <div className="grow" />
        <button className="btn" onClick={onClose}>
          Close
        </button>
      </div>
    </Modal>
  );
}
