# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — ORGANISM Windows .exe"""

import sys
from pathlib import Path

ROOT = Path(SPECPATH).resolve().parent

block_cipher = None

added_datas = [
    (str(ROOT / "organism" / "nursery" / "static"), "organism/nursery/static"),
    (str(ROOT / "organism" / "dna"), "organism/dna"),
    (str(ROOT / "data" / "dna"), "data/dna"),
]

hiddenimports = [
    "organism",
    "organism.nursery",
    "organism.nursery.server",
    "organism.nursery.download_info",
    "organism.autonomous.baby_agent",
    "organism.autonomous.baby_store",
    "organism.brain",
    "organism.brain.temporal_impulse_field",
    "organism.brain.resonance_templates",
    "organism.brain.impulse_field",
    "organism.brain.impulse_scaffold",
    "organism.brain.impulse_integration",
    "organism.cognition",
    "organism.dna",
    "organism.dna.interpreter",
    "organism.motor",
    "organism.sensory",
    "organism.learning",
    "organism.drives",
    "organism.teaching",
    "mind",
    "mind.types",
    "mindruntime",
    "mindruntime.gpu_engine",
    "mindruntime.gpu_core",
    "mindruntime.resonators",
    "mindruntime.dendritic_core",
    "mindruntime.gpu_physics_v2",
    "mindruntime.gpu_engine_v2",
    "mindruntime.field_v2",
    "yaml",
    "PIL",
    "PIL.Image",
    "numpy",
    "json",
    "http.server",
    "threading",
]

try:
    from PyInstaller.utils.hooks import collect_data_files, collect_submodules

    hiddenimports += collect_submodules("organism")
    hiddenimports += collect_submodules("mind")
    hiddenimports += collect_submodules("mindruntime")
    added_datas += collect_data_files("tzdata")
except Exception:
    pass

hiddenimports += ["tzdata", "zoneinfo"]

a = Analysis(
    [str(ROOT / "organism" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=added_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "packaging" / "pyi_rth_tzdata.py")],
    excludes=["tkinter", "matplotlib", "scipy", "torch", "pytest"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="ORGANISM-Windows",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
