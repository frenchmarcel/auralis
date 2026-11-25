import { api, errorMessage, type Material } from '../../api';
import type { useApp } from '../../AppContext';

type App = ReturnType<typeof useApp>;

/** Rename a custom material: only its name changes, its id (what simulations store) stays. */
export async function renameMaterial(app: App, m: Material): Promise<void> {
  const name = await app.prompt({ title: 'Rename material', label: 'Name', initial: m.name, confirmLabel: 'Rename' });
  if (!name || name === m.name) return;
  try {
    await api.renameMaterial(m.id, name);
    await app.reloadMaterials();
  } catch (e) {
    app.toast('error', errorMessage(e));
  }
}

/** Copy any material into My materials; resolves to the copy (null if cancelled or refused). */
export async function duplicateMaterial(app: App, m: Material, localOnly: boolean): Promise<Material | null> {
  const name = await app.prompt({
    title: 'Duplicate material',
    label: localOnly ? 'Name of the copy (it keeps the licence of the original: local use only, not redistributable)' : 'Name of the copy',
    initial: `${m.name} (copy)`,
    confirmLabel: 'Duplicate',
  });
  if (!name) return null;
  try {
    const copy = await api.duplicateMaterial(m.id, name);
    await app.reloadMaterials();
    app.toast('success', `“${copy.name}” added to My materials`);
    return copy;
  } catch (e) {
    app.toast('error', errorMessage(e));
    return null;
  }
}
