#!/usr/bin/env bash
# Start the Auralis web app: build the UI if needed, run the server, open the browser.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
PORT="${PORT:-8765}"
PY="$HERE/.venv/bin/python"

if [ ! -x "$PY" ]; then
    echo "Creating the app's Python environment (webapp/.venv)..."
    "${AURALIS_PYTHON:-python3}" -m venv --system-site-packages "$HERE/.venv"
    "$PY" -m pip install -q fastapi "uvicorn[standard]" python-multipart openpyxl ezdxf rhino3dm ifcopenshell h5py
fi
"$PY" -c "import openpyxl, ezdxf, rhino3dm, ifcopenshell, h5py" 2>/dev/null \
    || "$PY" -m pip install -q openpyxl ezdxf rhino3dm ifcopenshell h5py   # v0.5: xlsx, DXF/3DM/IFC, SOFA

if [ ! -f "$HERE/assets/visual/index.json" ]; then
    echo "Downloading the CC0 furniture models from Poly Haven (~75 MB, once)..."
    (cd "$REPO" && "$PY" -m webapp.server.fetch_assets) \
        || echo "WARNING: some visual models could not be downloaded; those objects are shown as their acoustic proxy."
fi
if [ ! -f "$HERE/assets/textures/index.json" ]; then
    echo "Downloading the CC0 surface textures and environment maps from Poly Haven (~41 MB, once)..."
    (cd "$REPO" && "$PY" -m webapp.server.fetch_textures) \
        || echo "WARNING: some textures could not be downloaded; the realistic view shows plain colours for them."
fi

if [ ! -f "$HERE/web/dist/index.html" ] || [ "${REBUILD:-0}" = 1 ]; then
    echo "Building the UI (webapp/web)..."
    [ -s "$HOME/.nvm/nvm.sh" ] && . "$HOME/.nvm/nvm.sh"
    (cd "$HERE/web" && { [ -d node_modules ] || npm install; } && npm run build)
fi

cd "$REPO"
"$PY" -m webapp.server --port "$PORT" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT INT TERM
for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null && break
    sleep 0.5
done
echo "Auralis is running at http://127.0.0.1:$PORT  (Ctrl+C to stop)"
command -v xdg-open >/dev/null && xdg-open "http://127.0.0.1:$PORT" >/dev/null 2>&1 || true
wait $SERVER
