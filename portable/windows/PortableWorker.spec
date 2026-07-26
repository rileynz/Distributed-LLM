# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

project = Path(SPECPATH).parent.parent
version_file = project / "packaging" / "windows" / "version_info.txt"
backend_root = project / "portable" / "runtime" / "backend"

datas = [
    (str(project / "portable" / "README.md"), "."),
    (str(project / "README.md"), "."),
    (str(project / "LICENSE"), "."),
    (str(project / "portable" / "Start-Portable-Worker.cmd"), "."),
]
if backend_root.exists():
    datas.append((str(backend_root), "backend"))

a = Analysis(
    [str(project / "portable" / "worker_entry.py")],
    pathex=[str(project)],
    binaries=[],
    datas=datas,
    hiddenimports=["cryptography"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DistributedLLM-Worker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    version=str(version_file),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="DistributedLLM-Portable-Worker",
)
