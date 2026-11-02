# AeroWatch AI — Windows Guide (Phase 10)

Target: Windows 10/11, ideally the owner's machine (RTX 3050 4 GB,
i7-12th-gen H, 16 GB RAM). Everything runs locally.

## Install (once)

```powershell
cd <project folder>
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
```

The script checks Python (≥ 3.10; 3.12 verified), creates `.venv`,
installs the pinned dependency lockfile (CPU build), detects an
NVIDIA GPU and *prints* — never silently runs — the CUDA swap
command, then runs `preflight.py`. **Internet is needed only for
this pip step.** Model weights are never downloaded: if
`weights\aerowatch_v7_balanced.pt` is missing, preflight fails with
that exact message.

GPU path (RTX 3050, optional — CPU works out of the box):

```powershell
.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python.exe evaluation\hardware_diagnostics.py   # confirm CUDA + VRAM
.venv\Scripts\python.exe evaluation\hardware_benchmark.py     # real measured numbers
```

## Launch (every time)

```powershell
powershell -ExecutionPolicy Bypass -File launch_windows.ps1            # normal
powershell -ExecutionPolicy Bypass -File launch_windows.ps1 -Offline   # network blocked
```

Opens at http://localhost:8501. Ctrl+C stops cleanly; history in
`runs\history` persists between runs.

## Verify the machine (owner commands)

| Purpose | Command |
|---|---|
| Environment + checkpoint integrity | `.venv\Scripts\python.exe preflight.py` |
| CPU/GPU/driver/CUDA report | `.venv\Scripts\python.exe evaluation\hardware_diagnostics.py` |
| Measured inference benchmark | `.venv\Scripts\python.exe evaluation\hardware_benchmark.py` |
| Physical camera validation | `.venv\Scripts\python.exe evaluation\camera_check.py --model` |
| History integrity | in-app: Detection History page, or `HistoryStore.verify_integrity()` |

## Offline operation

Runtime needs **no** internet: models are local files, history is a
local SQLite DB, and there are no external AI APIs. `-Offline`
(AEROWATCH_OFFLINE=1) additionally blocks every non-loopback socket
in the process, so an accidental external call fails loudly instead
of connecting. Only installation (pip) requires internet.

## Data, backup, recovery

- History DB: `runs\history\aerowatch_history.db` (metadata only —
  no images/videos/frames, no full paths, no operator IDs).
- **Backup:** Detection History → Retention settings → *Create
  backup now* (SQLite online backup, safe while running), or
  `HistoryStore.backup_to(path)` in Python.
- **Restore:** maintenance operation in Python —
  `HistoryStore(db).restore_from(backup_path, confirm=True)`. It
  validates integrity + schema first, and preserves the current DB
  as `*.pre-restore-<timestamp>.bak` before replacing anything.
- Retention defaults to 30 days (configurable in the same panel).

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Python not found" | Install Python 3.12 from python.org, tick *Add to PATH*, re-run setup |
| pip install fails | Check internet; re-run setup (safe to repeat) |
| Preflight: checkpoint FAIL | Copy the genuine v7 weights into `weights\`; a hash mismatch means the file is wrong/corrupt — do not bypass |
| App starts but slow | Expected on CPU for video; use the GPU path above, keep inference size 640 |
| Camera not found | Close other apps using the webcam, check Windows camera privacy settings, run `camera_check.py` |
| Port 8501 busy | Close the other Streamlit/app using it, or `streamlit run app.py --server.port 8502` |
| History looks empty | Confirm you are looking at the same machine/user; check `AEROWATCH_HISTORY_DIR` isn't pointing elsewhere |
