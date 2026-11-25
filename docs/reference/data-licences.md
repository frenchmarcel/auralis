# Data and licences

The Auralis code is under the MIT licence. Some data has its own licence.
Read this page before you share files or results.

| Data | Where | Licence | Notes |
|---|---|---|---|
| Auralis code | `auralis/`, `native/`, `webapp/`, `tools/` | MIT | |
| BRAS database | `data_bras/` | CC BY-SA 4.0 | Download it from TU Berlin (DepositOnce). Not in the repository. |
| Files derived from BRAS | `BRAS/` (geometry, manifests, reports) | CC BY-SA 4.0 | Converted geometry, materials and positions used by the bench and by the BRAS examples. See `BRAS/LICENSE.md`. The scripts are MIT. |
| FABIAN head | `data_hrtf/` | CC BY 4.0 | Credit TU Berlin (Brinkmann et al.) in any render that uses it. Auralis adds this credit for you. |
| Odeon library and recordings | `data_odeon/` | Odeon licence | Local research use only. Never share these files or put them in an archive. |
| Collected material data | `data_materials/` | Not settled | Local use only for now |
| Furniture models and textures | `webapp/assets/` | CC0 (Poly Haven) | Downloaded by `start.sh` |
| Paris street geometry | `webapp/server/data/examples/paris/` | ODbL | © OpenStreetMap contributors |
| Paris street recordings | `webapp/server/data/examples/paris/sounds/` | CC0 | From Freesound. Authors are credited in the renders. |

## Credits in renders

When the server renders a WAV file, it writes `credits.txt` next to it. The
file lists the credits that the licences ask for: the recordings, the measured
head, and the map data.

## What not to put in a shared archive

- `data_bras/`, `data_hrtf/`, `data_odeon/` and `data_materials/`
- `webapp/.venv/`, `webapp/web/node_modules/` and `webapp/assets/`
- generated outputs, like `BRAS/CR2/listening_out/`
